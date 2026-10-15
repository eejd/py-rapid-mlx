import Foundation
import ImageIO
import UniformTypeIdentifiers

struct ChatImageAttachment: Codable, Equatable, Hashable, Identifiable, Sendable {
    static let maxBytes = 20 * 1024 * 1024
    /// Per-message image budget, mirroring the document attachment budget
    /// (``ChatFileAttachment/maxAttachmentsPerMessage`` and friends). The
    /// per-file 20 MB cap is not enough on its own: a multi-select or drop can
    /// present dozens of individually valid images, and decoding every one and
    /// base64-encoding the whole set into a single request would freeze or
    /// exhaust the Desktop process. `maxImagesPerMessage` bounds the count and
    /// ``maxCombinedEncodedImageBytes`` bounds the aggregate **encoded** bytes.
    static let maxImagesPerMessage = 4
    /// Combined budget for the images in one message, measured in exact
    /// encoded `data:<mime>;base64,…` byte counts, not raw `Data.count`.
    /// Each image travels over the wire as a base64 data URL, so raw bytes
    /// under-count the request by ~4/3 and can deterministically blow past the
    /// engine's 8 MiB request-body cap with a 413. This budget stays below
    /// that cap while leaving dedicated room for text, history, tools, and
    /// JSON framing. **Value under review: Pixel (product).**
    static let maxCombinedEncodedImageBytes = 6 * 1024 * 1024

    /// A human-readable form of ``maxCombinedEncodedImageBytes`` for notices.
    static var formattedCombinedImageBudget: String {
        let bytes = maxCombinedEncodedImageBytes
        let mb = Double(bytes) / Double(1024 * 1024)
        if mb == mb.rounded() { return "\(Int(mb)) MB" }
        return String(format: "%.1f MB", mb)
    }

    /// User-facing explanation for attachments rejected by the per-message
    /// image budget. Keeping this copy beside the limits lets both the pre-read
    /// picker gate and the authoritative post-normalization gate report the
    /// same count and binding reason.
    static func budgetNotice(
        rejectedCount: Int = 0,
        limit: ImageBudgetLimit = .count
    ) -> String {
        let budget = formattedCombinedImageBudget
        let base = "Attach up to \(maxImagesPerMessage) images or \(budget) of images per message."
        guard rejectedCount > 0 else { return base }
        let reason: String = switch limit {
        case .count: "too many images"
        case .bytes: "their combined size exceeds the \(budget) budget"
        }
        let plural = rejectedCount == 1 ? "image was" : "images were"
        return "\(base) \(rejectedCount) \(plural) not added — \(reason)."
    }

    /// Why ``importCandidates(_:existingCount:existingBytes:)`` dropped a
    /// candidate. Kept distinct so a notice can tell the user whether the
    /// count or the combined-byte budget was the binding limit.
    enum ImageBudgetLimit {
        case count
        case bytes
    }

    /// Exact number of bytes a `data:<mime>;base64,<payload>` URL occupies for
    /// `rawBytes` of payload, without materialising the base64 string.
    /// `"data:"` + `<mime>` + `";base64,"` + base64(data); base64 of `n` bytes
    /// is `4 * ceil(n / 3)` and the integer `(n + 2) / 3 * 4` avoids floats.
    static func encodedDataURLByteCount(mimeType: String, rawBytes: Int) -> Int {
        let base64 = ((rawBytes + 2) / 3) * 4
        return 5 + mimeType.utf8.count + 8 + base64
    }
    /// Keep image preprocessing below the embedded engine's per-request
    /// vision-token budget. A 2048 × 1536 image is roughly 4K vision tokens
    /// on the recommended model, leaving useful margin below its 8K cap.
    static let maxVisionLongEdge = 2_048
    /// Bounds decoded memory before ImageIO creates a full bitmap. This still
    /// admits 48 MP iPhone captures while rejecting compressed image bombs.
    static let maxPixelCount = 64_000_000
    static let maxPixelDimension = 16_384

    static func dimensionsFit(width: Int, height: Int) -> Bool {
        width > 0
            && height > 0
            && width <= maxPixelDimension
            && height <= maxPixelDimension
            && width <= maxPixelCount / height
    }

    static func normalizedPixelSize(width: Int, height: Int) -> (width: Int, height: Int) {
        let longEdge = max(width, height)
        guard longEdge > maxVisionLongEdge else { return (width, height) }
        let scale = Double(maxVisionLongEdge) / Double(longEdge)
        return (
            max(1, Int((Double(width) * scale).rounded())),
            max(1, Int((Double(height) * scale).rounded()))
        )
    }

    /// Bound work before reading any selected image. The per-file 20 MB cap
    /// is enforced at read time; this pre-read gate stops the count and the
    /// aggregate bytes from growing unbounded across a multi-select/drop, which
    /// would otherwise decode and retain every image and base64 them all into a
    /// single request. Accepted order matches the selection; each candidate's
    /// on-disk file size is the pre-read estimate for the aggregate byte gate,
    /// charged at the encoded data-URL rate. The exact post-decode gate
    /// (``ChatImageAttachment/fittedForMessage(_:)``) is authoritative and is
    /// also applied at the wire, so a small pre-read skew cannot admit an
    /// over-budget request.
    static func importCandidates(
        _ urls: [URL],
        existingCount: Int,
        existingBytes: Int
    ) -> (accepted: [URL], rejectedCount: Int, limit: ImageBudgetLimit) {
        var accepted: [URL] = []
        var remainingCount = max(0, maxImagesPerMessage - max(0, existingCount))
        var remainingBytes = max(0, maxCombinedEncodedImageBytes - max(0, existingBytes))
        var limit: ImageBudgetLimit = .count
        for url in urls {
            guard remainingCount > 0 else { break }
            let size = (
                try? url.resourceValues(forKeys: [.fileSizeKey])
            )?.fileSize ?? 0
            // MIME is unknown before read; its prefix differs by one byte, which
            // is immaterial at this pre-read stage.
            let estimatedEncoded = encodedDataURLByteCount(mimeType: "image/jpeg", rawBytes: size)
            guard estimatedEncoded <= remainingBytes else {
                limit = .bytes
                continue
            }
            accepted.append(url)
            remainingCount -= 1
            remainingBytes -= estimatedEncoded
        }
        return (accepted, max(0, urls.count - accepted.count), limit)
    }

    /// Returns the ordered subset that fits the per-message image budget (count
    /// and combined bytes). Images cannot be truncated the way document text is
    /// (``ChatFileAttachment/fittedForMessage(_:)``), so an image that does not
    /// fit is dropped rather than reduced.
    static func fittedForMessage(
        _ attachments: [ChatImageAttachment]
    ) -> [ChatImageAttachment] {
        var result: [ChatImageAttachment] = []
        var remainingBytes = maxCombinedEncodedImageBytes
        for attachment in attachments {
            guard result.count < maxImagesPerMessage,
                  attachment.encodedDataURLByteCount <= remainingBytes else { continue }
            result.append(attachment)
            remainingBytes -= attachment.encodedDataURLByteCount
        }
        return result
    }

    let id: UUID
    let filename: String
    let mimeType: String
    let data: Data

    init(id: UUID = UUID(), filename: String, mimeType: String, data: Data) throws {
        guard data.count <= Self.maxBytes else { throw ValidationError.tooLarge }
        guard ["image/png", "image/jpeg", "image/gif"].contains(mimeType) else {
            throw ValidationError.unsupportedType
        }
        if mimeType == "image/gif",
           let source = CGImageSourceCreateWithData(data as CFData, nil),
           CGImageSourceGetCount(source) > 1 {
            throw ValidationError.animatedGIF
        }
        self.id = id
        self.filename = filename
        self.mimeType = mimeType
        self.data = data
    }

    init(contentsOf url: URL) throws {
        let values = try url.resourceValues(forKeys: [.fileSizeKey, .contentTypeKey])
        guard (values.fileSize ?? 0) <= Self.maxBytes else { throw ValidationError.tooLarge }
        guard values.contentType?.conforms(to: .image) == true,
              let source = CGImageSourceCreateWithURL(url as CFURL, nil)
        else { throw ValidationError.unsupportedType }
        guard CGImageSourceGetCount(source) == 1 else { throw ValidationError.animatedGIF }
        let sourceProperties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
            as? [CFString: Any] ?? [:]
        guard let width = (sourceProperties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
              let height = (sourceProperties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
              Self.dimensionsFit(width: width, height: height)
        else { throw ValidationError.tooManyPixels }

        let type = values.contentType
        let fitsVisionBudget = max(width, height) <= Self.maxVisionLongEdge
        if type?.conforms(to: .png) == true, fitsVisionBudget {
            try self.init(
                filename: url.lastPathComponent,
                mimeType: "image/png",
                data: Data(contentsOf: url)
            )
        } else if type?.conforms(to: .jpeg) == true, fitsVisionBudget {
            try self.init(
                filename: url.lastPathComponent,
                mimeType: "image/jpeg",
                data: Data(contentsOf: url)
            )
        } else if type?.conforms(to: .gif) == true, fitsVisionBudget {
            try self.init(
                filename: url.lastPathComponent,
                mimeType: "image/gif",
                data: Data(contentsOf: url)
            )
        } else {
            let normalized = try Self.normalizedStaticImage(
                source: source,
                sourceProperties: sourceProperties,
                at: url
            )
            try self.init(
                filename: normalized.filename,
                mimeType: normalized.mimeType,
                data: normalized.data
            )
        }
    }

    /// Normalize native still-image formats at the attachment boundary so the
    /// persisted attachment, preview, and wire request all share one truthful
    /// MIME/byte contract. Small PNG/JPEG/GIF files stay byte-for-byte
    /// unchanged above. Larger or native still images become JPEG, or PNG when
    /// alpha must be preserved. The ordinary initializer applies the same 20 MB
    /// wire cap to the normalized result.
    private static func normalizedStaticImage(
        source: CGImageSource,
        sourceProperties: [CFString: Any],
        at url: URL
    ) throws -> (filename: String, mimeType: String, data: Data) {
        let thumbnailOptions: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxVisionLongEdge,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(
            source,
            0,
            thumbnailOptions as CFDictionary
        ) else {
            throw ValidationError.unsupportedType
        }

        let preservesAlpha: Bool
        if let sourceHasAlpha = sourceProperties[kCGImagePropertyHasAlpha] as? NSNumber {
            preservesAlpha = sourceHasAlpha.boolValue
        } else {
            preservesAlpha = Self.containsTransparentPixel(image)
        }
        let targetType: UTType = preservesAlpha ? .png : .jpeg
        let mimeType = preservesAlpha ? "image/png" : "image/jpeg"
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            output,
            targetType.identifier as CFString,
            1,
            nil
        ) else { throw ValidationError.unsupportedType }

        var properties = sourceProperties
        // The thumbnail transform has already applied EXIF orientation.
        properties[kCGImagePropertyOrientation] = 1
        if !preservesAlpha {
            properties[kCGImageDestinationLossyCompressionQuality] = 0.9
        }
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else {
            throw ValidationError.unsupportedType
        }

        let base = url.deletingPathExtension().lastPathComponent
        let suffix = preservesAlpha ? "png" : "jpg"
        return ("\(base).\(suffix)", mimeType, output as Data)
    }

    /// Thumbnail backing stores may carry an alpha channel for an opaque
    /// source. Inspect the already-bounded thumbnail instead of treating the
    /// channel's mere presence as transparency.
    private static func containsTransparentPixel(_ image: CGImage) -> Bool {
        switch image.alphaInfo {
        case .none, .noneSkipFirst, .noneSkipLast: return false
        default: break
        }
        let bytesPerRow = image.width * 4
        var pixels = [UInt8](repeating: 255, count: bytesPerRow * image.height)
        guard let context = CGContext(
            data: &pixels,
            width: image.width,
            height: image.height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return true }
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return stride(from: 3, to: pixels.count, by: 4).contains { pixels[$0] < 255 }
    }

    var dataURL: String { "data:\(mimeType);base64,\(data.base64EncodedString())" }

    /// Exact number of bytes this attachment's `dataURL` occupies on the wire,
    /// without materialising `dataURL` just to measure it.
    var encodedDataURLByteCount: Int {
        Self.encodedDataURLByteCount(mimeType: mimeType, rawBytes: data.count)
    }

    enum ValidationError: LocalizedError {
        case tooLarge, tooManyPixels, unsupportedType, animatedGIF
        var errorDescription: String? {
            switch self {
            case .tooLarge: return "Images must be 20 MB or smaller."
            case .tooManyPixels: return "Images must be 64 megapixels or smaller."
            case .unsupportedType: return "Choose a supported still image."
            case .animatedGIF: return "Animated GIFs aren't supported."
            }
        }
    }
}

/// One chat message. Mirrors the OpenAI chat-completions schema closely
/// enough that the stream client can serialise an array of these directly
/// into the wire body.
///
/// ``id`` is local-only — used by SwiftUI for diffing and by the session
/// store for indexed mutation while a stream is in flight.
struct ChatMessage: Identifiable, Codable, Equatable, Hashable {
    enum Role: String, Codable, Sendable {
        case user
        case assistant
        case system
        /// Tool-result message. Carries ``toolCallID`` so the model
        /// can match the response to the call it asked for. Created
        /// programmatically by ``ChatViewModel`` during the tool
        /// round-trip; the user never types one directly.
        case tool
        /// Issue #477 forward-compatibility fallback. A ``sessions.json``
        /// written by a NEWER build (or hand-edited) can carry a role the
        /// current build doesn't know — e.g. a new OpenAI-schema role, or
        /// a downgrade after an auto-update. The custom ``init(from:)``
        /// below maps any unrecognised raw string here instead of
        /// throwing, so one forward-incompatible message no longer wipes
        /// the whole library. Rendered as a neutral system note in the UI
        /// and FILTERED OUT of the outbound wire body (serialising
        /// ``{"role":"unknown"}`` would 400 the next send). Encodes back
        /// to the stable ``"unknown"`` sentinel via the synthesised
        /// ``encode(to:)`` — the original raw string is intentionally not
        /// preserved (plain case, not ``unknown(String)``, so the
        /// compiler flags every exhaustive switch that must handle it).
        case unknown

        /// Forward-tolerant decode: an unrecognised raw string degrades
        /// to ``.unknown`` rather than throwing. Non-string values (a
        /// role encoded as a number, say) still throw — that element is
        /// then dropped by ``FailableDecodable`` one level up.
        init(from decoder: Decoder) throws {
            let raw = try decoder.singleValueContainer().decode(String.self)
            self = Role(rawValue: raw) ?? .unknown
        }
    }

    /// Streaming phase for assistant messages. User messages are always
    /// ``.complete`` from the moment they're created; system messages
    /// likewise. Only the assistant placeholder cycles through
    /// ``.streaming`` / ``.complete`` / ``.failed``.
    enum Status: String, Codable, Sendable {
        case complete
        case streaming
        case failed
        /// Issue #477 forward-compatibility fallback — same rationale as
        /// ``Role.unknown``. An unrecognised status from a newer / edited
        /// envelope degrades here instead of throwing. Treated exactly
        /// like ``.complete`` for all runtime purposes: it is NOT
        /// ``.streaming``, so ``SessionStore``'s streaming-count seeding
        /// and the typing-dot UI never wedge on a restored ``.unknown``
        /// row. Encodes back to the stable ``"unknown"`` sentinel.
        case unknown

        /// Forward-tolerant decode: an unrecognised raw string degrades
        /// to ``.unknown`` rather than throwing.
        init(from decoder: Decoder) throws {
            let raw = try decoder.singleValueContainer().decode(String.self)
            self = Status(rawValue: raw) ?? .unknown
        }
    }

    /// Whether a locally stored row belongs in the model-facing transcript.
    /// UI-authored affordances such as the onboarding welcome remain visible
    /// and persisted, but are not synthetic assistant turns on the wire.
    enum WireVisibility: String, Codable, Sendable {
        case model
        case transcriptOnly
    }

    /// Whether an image-bearing user turn was accepted by the model lane.
    /// Stored on the user turn because empty failed assistant rows are omitted
    /// from later wire history, while a rejected image must not poison every
    /// subsequent plain-text follow-up. A direct Retry still sends the image.
    enum ImageDeliveryStatus: String, Codable, Sendable {
        case pending
        /// One pre-token failure has already been allowed to retry through a
        /// later plain-text turn. Persisting this state on the image-bearing
        /// message keeps the retry budget tied to that exact turn.
        case retryable
        case accepted
        case rejected

        var permitsFollowUpInheritance: Bool {
            self == .retryable || self == .accepted
        }

        /// A newer outcome must fail closed for attachment inheritance rather
        /// than make one message undecodable and side the whole conversation
        /// as corrupt. The image remains visible and directly retryable.
        init(from decoder: Decoder) throws {
            let raw = try decoder.singleValueContainer().decode(String.self)
            self = ImageDeliveryStatus(rawValue: raw) ?? .rejected
        }
    }

    /// Messages written before ``wireVisibility`` shipped need one narrow
    /// migration: Quickstart's product-authored welcome was persisted as an
    /// ordinary assistant turn. Match only the two stable Rapid copy shapes;
    /// this is history-schema migration, not classification of user text.
    private static func isLegacyQuickstartWelcome(_ content: String) -> Bool {
        guard content.hasPrefix("You're chatting with ") else { return false }
        return content.hasSuffix(
            ", running entirely on your Mac. Open the picker any time to switch models."
        ) || (
            content.contains(" — a model picked so you can start chatting in about a minute. ")
                && content.hasSuffix("great first upgrade when you want more.")
        )
    }

    let id: UUID
    let role: Role
    /// The visible assistant prose / user prompt body. For Qwen3.5/3.6
    /// hybrid-thinking responses this excludes any ``reasoning_content``
    /// the model produced — that goes into ``reasoning`` so the UI can
    /// render it in a collapsed disclosure block.
    var content: String
    var imageAttachments: [ChatImageAttachment]
    var imageDeliveryStatus: ImageDeliveryStatus?
    /// Locally extracted document text. Kept separate from ``content`` so a
    /// multi-page source does not flood the transcript or copied user prose.
    var fileAttachments: [ChatFileAttachment]
    /// Hybrid-thinking trace (mlx-lm ``reasoning_content`` field). Only
    /// populated for assistant messages from hybrid models; empty string
    /// is treated as "no trace" by the UI.
    var reasoning: String
    var status: Status
    /// Optional inline error string for ``.failed`` rows. Shown under the
    /// (possibly partial) content with a red caption.
    var errorMessage: String?
    /// Rule-based diagnosis rendered instead of raw tool/transport details.
    /// Optional so sessions written by older builds decode unchanged.
    var failureKind: FailureDiagnosis.Kind?
    /// Tool calls returned by an assistant turn. ``nil`` for any other
    /// role and for assistant turns that produced plain text. The chat
    /// loop reads this to decide whether to run tools and continue.
    var toolCalls: [ToolCall]?
    /// Set on ``role == .tool`` messages — the ``ToolCall.id`` the
    /// content is responding to. Required by the OpenAI spec.
    var toolCallID: String?
    var gooseTool: GooseToolCard?
    /// v0.4.12: streaming-time + token-throughput stats. Populated
    /// at end-of-stream for assistant messages so the UI can show
    /// a small caption ("~84 tok/s · 2.4 s"). ``nil`` for all
    /// older sessions decoded from disk — the custom init below
    /// defaults it on missing-key, mirroring the pre-existing
    /// pattern for ``ChatSession.isPinned``.
    var stats: MessageStats?
    /// Cycle-2 (2026-06-19) F-002 marker: set to ``true`` ONLY when a
    /// reasoning model exhausted its ``max_tokens`` budget mid-think,
    /// producing empty ``content`` + populated ``reasoning`` +
    /// ``finish_reason: "length"`` (see
    /// ``ChatViewModel.TerminalOutcome.reasoningOnlyTruncated``).
    /// The chat view keys on THIS flag (not on
    /// ``content.isEmpty && !reasoning.isEmpty``) to decide whether
    /// to auto-expand the reasoning disclosure, relabel it as
    /// "Thinking trace (cut off)", and route the VoiceOver
    /// accessibility caption — because a user-cancelled stream and a
    /// chat-template parser bug can ALSO land with empty content +
    /// populated reasoning, and they need different UX.
    ///
    /// Defaults to ``false`` everywhere. Old on-disk sessions decoded
    /// before the cycle-2 release have no key for this field; the
    /// custom ``init(from:)`` below uses ``decodeIfPresent`` with a
    /// ``false`` fallback so they load cleanly. (Swift's synthesised
    /// ``Decodable`` would have thrown on a missing non-optional
    /// ``Bool``; codex r1 NIT clarified.)
    var reasoningTruncated: Bool
    /// Cycle-13 (2026-06-20) F-5 marker: set to ``true`` ONLY when a
    /// non-reasoning assistant turn exhausted its ``max_tokens`` budget
    /// mid-answer — i.e. ``finish_reason == "length"`` AND ``content``
    /// is non-empty AND ``reasoning`` is empty (see
    /// ``ChatViewModel.runOneStream``'s ``.finished`` handler).
    /// Verbose-output dense models (nemotron-30b-4bit and similar)
    /// would emit a 200-token LaTeX derivation for "what is 17*23?"
    /// against the default 200-token cap and the row would render as
    /// a normal completed answer with no indication the model was
    /// cut off — so the user reads a half-finished derivation as the
    /// real reply.
    ///
    /// The chat view keys on THIS flag (not on ``finish_reason ==
    /// "length"`` directly) to decide whether to paint a subtle
    /// "Answer cut off (Max Tokens hit). Increase Max Tokens to see
    /// the rest." caption inline at the bottom of the bubble.
    ///
    /// Disjoint from ``reasoningTruncated``:
    ///   * ``reasoningTruncated`` — empty content + populated
    ///     reasoning + length (PR #317's reasoning-only fallback).
    ///   * ``contentTruncated`` — populated content + empty reasoning
    ///     + length (this cycle-13 verbose-output fix).
    ///   * Both false on a content + reasoning + length shape, since
    ///     ``classifyTerminal`` already treats that as a real
    ///     completion (the answer body landed). See
    ///     ``ChatViewVerboseOutputBadgeTests`` for the 4-cell truth
    ///     table.
    ///
    /// Defaults to ``false`` everywhere. Old on-disk sessions decoded
    /// before this cycle's release have no key for this field; the
    /// custom ``init(from:)`` below uses ``decodeIfPresent`` with a
    /// ``false`` fallback so they load cleanly (same back-compat
    /// shim used for ``reasoningTruncated``).
    var contentTruncated: Bool
    /// Issue #308 (2026-06-20) marker: set to ``true`` ONLY when an
    /// assistant turn met every gate in ``shouldFlagToolNotCalled``
    /// — tools were sent in the request body, zero ``tool_calls``
    /// were emitted, prose body looks like a raw / numeric answer,
    /// AND the user's prompt looked calculator- / search-shaped.
    /// The chat view paints a lightweight dismissible caption above
    /// the bubble ("This model didn't call a tool — verify the
    /// answer.") so the user isn't silently misled by a small
    /// model's hallucinated calculation.
    ///
    /// Defaults to ``false`` everywhere. Old on-disk sessions
    /// decoded before this release have no key for this field; the
    /// custom ``init(from:)`` below uses ``decodeIfPresent`` with a
    /// ``false`` fallback so they load cleanly (same back-compat
    /// shim used for ``reasoningTruncated`` /
    /// ``contentTruncated``).
    var toolNotCalledFlagged: Bool
    /// Issue #513 marker (defense-in-depth, layer 3): set to ``true``
    /// ONLY when a finished assistant turn had tools advertised on the
    /// request, fired **zero** ``tool_calls``, AND its ``content`` is
    /// essentially just a malformed tool-call artifact the engine parser
    /// couldn't recover (a raw ``<tool_call>`` / ``<parameter=`` /
    /// ``[TOOL_CALLS]`` / ``<｜tool▁calls▁begin｜>`` fragment or a bare
    /// tool-call-shaped JSON object). The chat view then replaces the
    /// raw dump with a quiet "the model tried to use a tool but its
    /// request couldn't be read" caption instead of rendering the
    /// envelope — so a rare per-turn glitch on an otherwise-good model
    /// doesn't surface confusing machine syntax to the user. See
    /// ``shouldSuppressToolCallArtifact``.
    ///
    /// Defaults to ``false`` everywhere. Old on-disk sessions decoded
    /// before this release have no key for this field; the custom
    /// ``init(from:)`` below uses ``decodeIfPresent`` with a ``false``
    /// fallback so they load cleanly (same back-compat shim used for
    /// ``reasoningTruncated`` / ``contentTruncated`` /
    /// ``toolNotCalledFlagged``).
    var toolCallArtifactSuppressed: Bool
    var wireVisibility: WireVisibility
    /// Parent node in the conversation tree, or ``nil`` for a root turn.
    ///
    /// A conversation stores EVERY branch, not just the visible transcript;
    /// the path on screen is derived by walking up from
    /// ``ChatConversation.activeLeafID`` through this link (see
    /// ``ChatConversation/activePath``). Two messages sharing a ``parentID``
    /// are siblings — alternative continuations of the same point — which is
    /// what Regenerate / Retry / editing a prompt now produce instead of
    /// truncating the tail away.
    ///
    /// Siblings are deliberately NOT mirrored in a `childrenIDs` array on the
    /// parent: one authoritative edge per node cannot drift out of sync,
    /// whereas a bidirectional pair has to be repaired on every insert,
    /// delete, and decode. Sibling order is derived from ``createdAt``.
    ///
    /// Optional, and absent from every conversation written before branching
    /// shipped. ``ChatConversation``'s decode reconnects such a legacy linear
    /// array into a degenerate tree (each row parented to the one before it),
    /// so old transcripts render exactly as they always did.
    var parentID: UUID?
    let createdAt: Date

    /// Wire-only trailer appended after this row's prose AND its attachment
    /// extracts — see ``modelContent``.
    ///
    /// Deliberately absent from ``CodingKeys``: it is set on the throwaway
    /// array ``ChatViewModel`` builds for one request, never on the
    /// transcript, so it is not part of the conversation and must not reach
    /// `conversations.json`. ``init(from:)`` therefore restores it as `nil`.
    ///
    /// The one producer is ``ChatViewModel/stampingClockContext(on:calendar:)``,
    /// which needs each user turn's wall clock to land after that turn's
    /// attachment extracts. Writing it into ``content`` instead would put it
    /// in FRONT of the extract, so the first request carrying a document and
    /// the next one would diverge before the document rather than after it,
    /// and the engine's prefix cache could not reuse the document.
    var wireSuffix: String?

    init(
        id: UUID = UUID(),
        role: Role,
        content: String = "",
        imageAttachments: [ChatImageAttachment] = [],
        imageDeliveryStatus: ImageDeliveryStatus? = nil,
        fileAttachments: [ChatFileAttachment] = [],
        reasoning: String = "",
        status: Status = .complete,
        errorMessage: String? = nil,
        failureKind: FailureDiagnosis.Kind? = nil,
        toolCalls: [ToolCall]? = nil,
        toolCallID: String? = nil,
        gooseTool: GooseToolCard? = nil,
        stats: MessageStats? = nil,
        reasoningTruncated: Bool = false,
        contentTruncated: Bool = false,
        toolNotCalledFlagged: Bool = false,
        toolCallArtifactSuppressed: Bool = false,
        wireVisibility: WireVisibility = .model,
        parentID: UUID? = nil,
        createdAt: Date = Date(),
        wireSuffix: String? = nil
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.imageAttachments = imageAttachments
        self.imageDeliveryStatus = imageDeliveryStatus
        self.fileAttachments = fileAttachments
        self.reasoning = reasoning
        self.status = status
        self.errorMessage = errorMessage
        self.failureKind = failureKind
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
        self.gooseTool = gooseTool
        self.stats = stats
        self.reasoningTruncated = reasoningTruncated
        self.contentTruncated = contentTruncated
        self.toolNotCalledFlagged = toolNotCalledFlagged
        self.toolCallArtifactSuppressed = toolCallArtifactSuppressed
        self.wireVisibility = wireVisibility
        self.parentID = parentID
        self.createdAt = createdAt
        self.wireSuffix = wireSuffix
    }

    /// Codex r1 MAJOR-1: keep ``reasoningTruncated`` decodable from
    /// pre-cycle-2 session envelopes (the on-disk JSON has no such
    /// key). Swift's synthesised init(from:) throws on a missing
    /// non-optional, so the custom init below falls back to ``false``
    /// for that one key and defers all other fields to the standard
    /// container shape.
    enum CodingKeys: String, CodingKey {
        case id, role, content, imageAttachments, imageDeliveryStatus
        case fileAttachments, reasoning, status
        case errorMessage, failureKind, toolCalls, toolCallID, gooseTool
        case stats, reasoningTruncated, contentTruncated
        case toolNotCalledFlagged
        case toolCallArtifactSuppressed
        case wireVisibility
        case parentID
        case createdAt
        /// The outcome as THIS build understands it. ``failureKind`` carries
        /// the same outcome narrowed to a value older builds can decode — see
        /// ``encode(to:)``.
        case failureKindV2
    }

    /// Hand-written so ``failureKind`` is persisted TWICE.
    ///
    /// ``FailureDiagnosis.Kind`` decodes strictly in every build already in
    /// users' hands, and ``ConversationStore.load`` turns ONE undecodable
    /// message into "the whole history is corrupt": the file is sided to
    /// `conversations.corrupt-<uuid>.json` and the sidebar comes up empty.
    /// So a raw value added after a release must never be written into the
    /// key those builds read — that would cost a user who downgrades their
    /// entire visible history, which is precisely the failure ``Role.unknown``
    /// and ``Status.unknown`` exist to prevent on the other two enums.
    ///
    /// Hence: ``failureKind`` gets ``legacyPersistedKind`` (always a value
    /// every shipped build knows, so an old build reads a slightly coarser
    /// outcome and keeps the conversation), and ``failureKindV2`` — a key old
    /// builds simply ignore — gets the real one. Adding a future kind needs
    /// nothing here beyond its ``legacyPersistedKind`` mapping.
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(role, forKey: .role)
        try c.encode(content, forKey: .content)
        try c.encode(imageAttachments, forKey: .imageAttachments)
        try c.encodeIfPresent(imageDeliveryStatus, forKey: .imageDeliveryStatus)
        try c.encode(fileAttachments, forKey: .fileAttachments)
        try c.encode(reasoning, forKey: .reasoning)
        try c.encode(status, forKey: .status)
        try c.encodeIfPresent(errorMessage, forKey: .errorMessage)
        try c.encodeIfPresent(failureKind?.legacyPersistedKind, forKey: .failureKind)
        try c.encodeIfPresent(failureKind, forKey: .failureKindV2)
        try c.encodeIfPresent(toolCalls, forKey: .toolCalls)
        try c.encodeIfPresent(toolCallID, forKey: .toolCallID)
        try c.encodeIfPresent(gooseTool, forKey: .gooseTool)
        try c.encodeIfPresent(stats, forKey: .stats)
        try c.encode(reasoningTruncated, forKey: .reasoningTruncated)
        try c.encode(contentTruncated, forKey: .contentTruncated)
        try c.encode(toolNotCalledFlagged, forKey: .toolNotCalledFlagged)
        try c.encode(toolCallArtifactSuppressed, forKey: .toolCallArtifactSuppressed)
        try c.encode(wireVisibility, forKey: .wireVisibility)
        // Omitted for root turns. Non-root rows always carry it — including
        // in conversations that never branched — which is safe because it is
        // an additive key every shipped decoder ignores; the conversation-
        // level schema marker is the `branches` key, not this one.
        try c.encodeIfPresent(parentID, forKey: .parentID)
        try c.encode(createdAt, forKey: .createdAt)
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(UUID.self, forKey: .id)
        self.role = try c.decode(Role.self, forKey: .role)
        self.content = try c.decode(String.self, forKey: .content)
        self.imageAttachments = try c.decodeIfPresent([ChatImageAttachment].self, forKey: .imageAttachments) ?? []
        self.imageDeliveryStatus = try c.decodeIfPresent(
            ImageDeliveryStatus.self,
            forKey: .imageDeliveryStatus
        )
        self.fileAttachments = try c.decodeIfPresent([ChatFileAttachment].self, forKey: .fileAttachments) ?? []
        self.reasoning = try c.decode(String.self, forKey: .reasoning)
        self.status = try c.decode(Status.self, forKey: .status)
        self.errorMessage = try c.decodeIfPresent(String.self, forKey: .errorMessage)
        // Prefer the finer v2 value; fall back to the original key for rows
        // written before it existed (and for rows an older build re-saved).
        //
        // Read v2 as a raw string rather than through ``Kind``'s tolerant
        // decode: a value this build doesn't recognise came from a NEWER one,
        // which wrote its closest known ancestor into ``failureKind`` — and
        // that is strictly better than the blanket degrade to ``.toolFailed``.
        let modernRaw = try c.decodeIfPresent(String.self, forKey: .failureKindV2)
        let legacyKind = try c.decodeIfPresent(FailureDiagnosis.Kind.self, forKey: .failureKind)
        self.failureKind = modernRaw.flatMap(FailureDiagnosis.Kind.init(rawValue:)) ?? legacyKind
        self.toolCalls = try c.decodeIfPresent([ToolCall].self, forKey: .toolCalls)
        self.toolCallID = try c.decodeIfPresent(String.self, forKey: .toolCallID)
        self.gooseTool = try c.decodeIfPresent(GooseToolCard.self, forKey: .gooseTool)
        self.stats = try c.decodeIfPresent(MessageStats.self, forKey: .stats)
        self.reasoningTruncated = try c.decodeIfPresent(Bool.self, forKey: .reasoningTruncated) ?? false
        // Cycle-13 (2026-06-20) F-5: same back-compat shim as
        // ``reasoningTruncated`` — sessions saved before this cycle's
        // release have no key, so we default to false.
        self.contentTruncated = try c.decodeIfPresent(Bool.self, forKey: .contentTruncated) ?? false
        // Issue #308 (2026-06-20): same back-compat shim — sessions
        // saved before this release have no key for this field; we
        // default to false so old transcripts decode cleanly.
        self.toolNotCalledFlagged = try c.decodeIfPresent(Bool.self, forKey: .toolNotCalledFlagged) ?? false
        // Issue #513: same back-compat shim — sessions saved before this
        // release have no key for this field; default to false so old
        // transcripts decode cleanly.
        self.toolCallArtifactSuppressed = try c.decodeIfPresent(Bool.self, forKey: .toolCallArtifactSuppressed) ?? false
        if let storedVisibility = try c.decodeIfPresent(WireVisibility.self, forKey: .wireVisibility) {
            self.wireVisibility = storedVisibility
        } else {
            self.wireVisibility = role == .assistant && Self.isLegacyQuickstartWelcome(content)
                ? .transcriptOnly
                : .model
        }
        // Absent on every row written before branching shipped, and absent on
        // root turns thereafter. ``ChatConversation``'s decode rebuilds the
        // parent chain for a legacy linear array, so ``nil`` here is not
        // assumed to mean "root" until that repair has run.
        self.parentID = try c.decodeIfPresent(UUID.self, forKey: .parentID)
        self.createdAt = try c.decode(Date.self, forKey: .createdAt)
        // Transient by construction — a persisted turn carries no wire
        // trailer, and the next request mints a fresh one.
        self.wireSuffix = nil
    }

    /// Text sent to the model. Document extracts stay out of the visible
    /// ``content`` property but remain part of this turn on every retry and
    /// follow-up request.
    /// ``wireSuffix`` is joined LAST, after the attachment extracts, so a
    /// per-request trailer cannot displace the document text that the
    /// engine's prefix cache needs to find unchanged.
    var modelContent: String {
        let trailer = wireSuffix.flatMap {
            $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : [$0]
        } ?? []
        guard !fileAttachments.isEmpty else {
            guard !trailer.isEmpty else { return content }
            // An empty prose row must not gain a leading blank line.
            let head = content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? [] : [content]
            return (head + trailer).joined(separator: "\n\n")
        }
        let request = content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "Analyze the attached file and summarize the important findings."
            : content
        return ([request] + fileAttachments.map(\.promptText) + trailer)
            .joined(separator: "\n\n")
    }

    // MARK: - Cycle-13 (2026-06-20) F-5: length-truncation badge

    /// Returns ``true`` when a non-reasoning assistant turn was cut
    /// off by ``max_tokens`` mid-answer — i.e. an answer body exists
    /// (or at least started) but the model never reached its natural
    /// stop token. The chat view paints a subtle inline caption
    /// ("Answer cut off (Max Tokens hit). Increase Max Tokens to see
    /// the rest.") under the bubble for this shape so the user
    /// understands the half-finished body isn't a real answer.
    ///
    /// ## Gates
    ///
    ///   * ``finish_reason == "length"`` — the server explicitly told
    ///     us the cap was hit. Without this gate every short answer
    ///     a verbose model produced would falsely wear the badge.
    ///   * ``content`` non-empty after whitespace trim — there's an
    ///     answer body to be "cut off". The empty-content + length
    ///     shape is owned by ``classifyTerminal`` (it routes to either
    ///     ``.reasoningOnlyTruncated`` or ``.emptyTurnFailure``); the
    ///     badge path stays out of that lane.
    ///   * ``reasoning`` empty after whitespace trim — a populated
    ///     reasoning trace on a length-truncated turn is PR #317's
    ///     domain (the reasoning-only fallback owns the
    ///     ``reasoningTruncated`` flag + auto-expanded disclosure
    ///     copy). The badge path stays out of that lane too, so a
    ///     reasoning model that hit the cap mid-answer-after-thinking
    ///     gets the reasoning fallback's copy + auto-expand, not a
    ///     redundant length-truncated badge.
    ///
    /// Role-agnostic so unit tests can exercise the gates directly;
    /// the view layer enforces "only paint on assistant rows".
    ///
    /// See ``ChatViewVerboseOutputBadgeTests`` for the 4-cell truth
    /// table covering (finish_reason ∈ {.length, .stop}) ×
    /// (reasoning ∈ {empty, non-empty}).
    static func shouldFlagContentTruncated(
        content: String,
        reasoning: String,
        finishReason: String?
    ) -> Bool {
        guard finishReason == "length" else { return false }
        let contentEmpty = content
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty
        guard !contentEmpty else { return false }
        let reasoningEmpty = reasoning
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty
        guard reasoningEmpty else { return false }
        return true
    }

    /// Cycle-13 F-5 (2026-06-20): pinned visible copy for the
    /// length-truncation badge. Lifted onto ``ChatMessage`` (the model
    /// layer) instead of ``MessageRow`` (the view layer) because
    /// ``MessageRow`` is ``private`` to ``ChatView.swift`` and the
    /// test suite needs to snapshot the exact string an accidental
    /// reword would change. ``MessageRow`` reads through this
    /// constant so the painted string and the snapshotted string
    /// can never drift apart.
    ///
    /// Copy reference: ChatGPT-Desktop's "Continue generating" pattern.
    /// We surface (i) what happened ("Answer cut off") and (ii) the
    /// user-facing knob to raise ("Max Tokens"). See
    /// ``ChatViewVerboseOutputBadgeTests.badgeCopySnapshot``.
    static let lengthTruncationBadgeCopy: String =
        "Answer cut off (Max Tokens hit). Increase Max Tokens to see the rest."

    /// Cycle-13 F-5 (2026-06-20): VoiceOver caption for the
    /// length-truncation badge. Reads the same information as the
    /// visible text but as a single well-formed sentence so the screen
    /// reader's pacing doesn't break on the parenthetical. Pinned
    /// alongside the visible copy so an accessibility regression
    /// surfaces in the tests, not silently in production.
    static let lengthTruncationBadgeAccessibilityLabel: String =
        "Answer cut off because the Max Tokens limit was hit. Increase Max Tokens in Settings to see the rest of the answer."


}

/// End-of-stream summary stats for an assistant turn. Carried on
/// the message itself so the rendered "~84 tok/s · 2.4 s" caption
/// survives session reload — recomputing from ``createdAt`` would
/// be wrong after a reload (the cold-start latency leaks in).
///
/// ``charCount`` is the only field we can reliably populate today:
/// neither rapid-mlx nor the OpenAI streaming spec emits per-chunk
/// usage by default, and ``stream_options.include_usage`` is on
/// the v0.4.13 backlog. We use char count as a proxy ("~" prefix
/// in the UI signals estimate, not authoritative). When the
/// include_usage wiring lands, ``promptTokens`` / ``completionTokens``
/// will become populated and the UI will drop the tilde.
struct MessageStats: Codable, Equatable, Hashable {
    /// Wall-clock seconds from the assistant placeholder being
    /// inserted to the [DONE] event arriving. Doesn't include
    /// cold-start (the model was already loaded by then).
    var elapsedSeconds: Double
    /// Characters in ``content`` at end-of-stream. Estimating
    /// tokens at ~4 chars/token gets us roughly within 15% on
    /// English; non-English / code is noisier but still useful.
    var charCount: Int
    /// Server-reported prompt token count from the final
    /// stream chunk's ``usage`` block. ``nil`` until v0.4.13
    /// wires ``stream_options.include_usage``.
    var promptTokens: Int?
    /// Server-reported completion token count. ``nil`` until
    /// v0.4.13.
    var completionTokens: Int?
    /// Seconds from dispatching the request to the FIRST content token
    /// arriving — the prompt-processing (prefill) half of the turn.
    ///
    /// ``nil`` on transcripts persisted before this field existed and on
    /// turns that produced no content, in which case every rate below
    /// degrades to the pre-existing whole-turn arithmetic.
    var timeToFirstTokenSeconds: Double?
    /// Did this turn emit a reasoning trace?
    ///
    /// Only the char-count estimate needs it, to know that its numerator
    /// (visible prose) and its denominator (a window that opened on the
    /// first reasoning token) are not describing the same generation.
    /// Optional for the same reason as ``timeToFirstTokenSeconds``: older
    /// transcripts have no such field and decode as ``nil``.
    var reasoningEmitted: Bool?

    /// ``reasoningEmitted`` with the legacy default. A transcript written
    /// before the field existed is treated as prose-only, which is what
    /// those rows were captioned as at the time.
    var emittedReasoning: Bool { reasoningEmitted ?? false }

    /// Seconds spent generating — everything after the first token landed.
    ///
    /// This, not ``elapsedSeconds``, is the denominator for a throughput
    /// number. Dividing by the whole turn folds prefill into "tok/s" and
    /// makes the answer's LENGTH the dominant term: the desktop advertised
    /// ~61 tok/s for `qwen3.5-4b-4bit` while the same model, in the same
    /// second, captioned a short reply at 13 tok/s — prefill of a
    /// ~950-token tool-carrying prompt was 93 % of that turn. Same
    /// machine, same model, two numbers 5x apart, both labelled "tok/s".
    /// The recorded TTFT, but only when it describes an interval that can
    /// exist: inside the turn it belongs to, and not negative.
    ///
    /// A persisted transcript can carry nonsense here — a hand-edited
    /// session file, or a row written by a build that measured this with a
    /// wall clock rather than the monotonic one used now — and a value at
    /// or past ``elapsedSeconds`` is not a prefill measurement.
    /// One accessor so arithmetic and presentation can never disagree
    /// about which values are real: rejecting it for the rate while still
    /// rendering "1.2 s to first token · 1.0 s" would just move the lie
    /// into the caption.
    var validTimeToFirstToken: Double? {
        guard let timeToFirstTokenSeconds,
              timeToFirstTokenSeconds >= 0,
              timeToFirstTokenSeconds < elapsedSeconds else { return nil }
        return timeToFirstTokenSeconds
    }

    /// Whether this row came from a build that measures prefill at all.
    ///
    /// The field's PRESENCE is the discriminator, not its validity, and the
    /// distinction decides what an unusable TTFT falls back to. A transcript
    /// persisted before ``timeToFirstTokenSeconds`` existed decodes as `nil`
    /// here and is legitimately captioned with the whole-turn arithmetic it
    /// was first shown with — there is no better measurement to be had, and
    /// re-rendering history is its own kind of wrong.
    ///
    /// A row that carries a value the guards then REJECT is a different
    /// animal: a corrupt measurement from a build that should have produced
    /// a good one. Falling back to the whole turn *there* would reintroduce
    /// the plausible-but-wrong number this change exists to remove, on
    /// precisely the rows already known to be untrustworthy, and it would be
    /// indistinguishable from a real reading. Those get no rate at all.
    var measuresPrefill: Bool { timeToFirstTokenSeconds != nil }

    var decodeSeconds: Double? {
        guard let ttft = validTimeToFirstToken else {
            return measuresPrefill ? nil : elapsedSeconds
        }
        let decode = elapsedSeconds - ttft
        return decode > 0 ? decode : nil
    }

    /// Heuristic tokens/sec from char count, for a server that reports no
    /// usage at all. UI prefixes with "~" to signal estimate.
    ///
    /// Gated on ``completionTokens == nil`` so it is a fallback for
    /// *missing* data and never a second opinion that overrides a
    /// deliberate ``nil`` from ``reportedTokensPerSecond``. Without that
    /// gate a one-token reply — which the reported path declines to rate
    /// because there is no interval to divide by — fell straight through
    /// to this estimate and printed "~1 tok/s" anyway.
    ///
    /// Suppressed on a turn that emitted reasoning, because the two halves
    /// of the fraction would then measure different things: the decode
    /// window opens at the first token on ANY channel (a reasoning model's
    /// trace comes first), while ``charCount`` counts only the visible
    /// prose. A long think followed by one short sentence would divide a
    /// handful of characters by the whole reasoning window and understate
    /// the rate by however long the model thought — reintroducing, in the
    /// fallback path, the exact distortion this change removed from the
    /// main one. No number beats a wrong one; the caption still carries
    /// time-to-first-token and the total.
    /// The estimate carries the same inverse-TPOT shape as the reported
    /// path, and for the same reason: when there IS a measured prefill, the
    /// first token is what ended it, so it was not produced inside
    /// ``decodeSeconds`` and does not belong in the numerator. Leaving the
    /// subtraction out here would make a four-character reply — one
    /// estimated token, zero token intervals — divide by a window it never
    /// occupied and print a rate, which is exactly what the reported path
    /// declines to do one accessor below.
    var estimatedTokensPerSecond: Double? {
        guard completionTokens == nil, !emittedReasoning else { return nil }
        guard let decodeSeconds, decodeSeconds > 0.05 else { return nil }  // < 50 ms is noise
        let estTokens = Double(charCount) / 4.0
        guard validTimeToFirstToken != nil else {
            return estTokens / decodeSeconds
        }
        guard estTokens > 1 else { return nil }
        return (estTokens - 1) / decodeSeconds
    }

    /// Authoritative tokens/sec when the server reported usage.
    ///
    /// With a TTFT, ``completionTokens - 1`` intervals span the decode
    /// window: the first token is what ENDS prefill, so it is not produced
    /// during ``decodeSeconds``. That is the standard inverse-TPOT
    /// definition, and it needs at least two tokens to mean anything — a
    /// one-token reply returns ``nil`` and the caption omits a rate rather
    /// than dividing by a window the token never occupied.
    ///
    /// **Without** a TTFT the subtraction has nothing to stand on: the
    /// denominator is the whole turn, which begins before the first token,
    /// so all N tokens fall inside it. Transcripts written before this
    /// field existed therefore keep the exact arithmetic they were
    /// captioned with — `N / elapsed` — rather than silently shifting to
    /// `(N - 1) / elapsed` and re-rendering history slightly differently
    /// than it was first shown. That branch is reachable *only* for those
    /// legacy rows: see ``measuresPrefill``, which withholds
    /// ``decodeSeconds`` entirely from a modern row whose stamp is corrupt,
    /// so this accessor returns `nil` there instead of falling through to
    /// the whole-turn number.
    var reportedTokensPerSecond: Double? {
        guard let completionTokens, completionTokens > 0,
              let decodeSeconds, decodeSeconds > 0.05 else { return nil }
        guard validTimeToFirstToken != nil else {
            return Double(completionTokens) / decodeSeconds
        }
        guard completionTokens > 1 else { return nil }
        return Double(completionTokens - 1) / decodeSeconds
    }
}

extension Duration {
    /// Whole plus fractional seconds, for storing an interval in the
    /// `Double` fields of ``MessageStats``.
    ///
    /// The durations here come from ``ContinuousClock`` rather than
    /// differences of `Date`, because both numbers this feeds are the
    /// *difference* between two readings taken seconds apart. A wall clock
    /// can be stepped by NTP or by the user between those two reads, and a
    /// step of a few hundred milliseconds is enough to turn a real decode
    /// window into a nonsensical one while still landing inside every
    /// range guard — a silently wrong rate, which is the failure mode this
    /// whole change set exists to eliminate. A monotonic clock cannot be
    /// stepped, so that class of corruption stops being possible rather
    /// than being detected after the fact.
    var seconds: Double {
        let parts = components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}
