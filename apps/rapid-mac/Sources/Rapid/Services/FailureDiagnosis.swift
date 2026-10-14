import Foundation

/// A user-facing explanation for a known outcome that stopped short of a
/// result. Raw process, transport, and tool output stays in logs/model
/// context; views render only this stable copy and, when available, one
/// concrete recovery action.
///
/// Most kinds describe a genuine fault. One does not: ``Kind/userDeclined``
/// is the user answering "no" to a permission prompt, which is an ordinary
/// outcome and must never be dressed as a malfunction — see ``Severity``.
struct FailureDiagnosis: Equatable, Sendable {
    enum Kind: String, CaseIterable, Codable, Equatable, Hashable, Sendable {
        case modelOutOfMemory
        case modelLoadFailed
        case engineNotRunning
        case webSearchOffline
        case webSearchUnavailable
        /// A user-supplied Keenable key was rejected. Keep keyed account
        /// failures distinct from provider availability: the network is up,
        /// retrying the same request cannot repair the credential, and the
        /// useful destination is the key field in Settings.
        case webSearchKeyRejected
        /// The stored Keenable key has exhausted its monthly credit allowance.
        case webSearchKeyQuotaExceeded
        /// The stored Keenable key hit its account/organisation rate cap.
        case webSearchKeyRateLimited
        /// The free DuckDuckGo backend throttled this machine. Distinct from
        /// ``webSearchUnavailable`` because the remedy is different: nothing in
        /// Settings is misconfigured, so "check its settings" sends the user to
        /// a dead end. The only real fix is a different backend.
        case webSearchRateLimited
        /// A browse fetch crossed the bounded response cap. The agent can
        /// normally recover by searching or choosing a narrower page.
        case browsePageTooLarge
        case commandPermissionDenied
        case commandFailed
        case fileNotFound
        case filePermissionDenied
        case toolFailed
        case userDeclined
        case downloadFailed
        /// The user stopped the pull themselves. Distinct from
        /// ``downloadFailed`` because the two differ in the only thing the
        /// copy is for: nothing went wrong, and "check your connection" is
        /// advice about a fault that did not happen. Paper 05.1 flags the
        /// collapsed reading explicitly ("For a user-initiated stop that
        /// connection advice is wrong … the fix belongs in FailureDiagnoser,
        /// not in the design"), so the split lives here rather than in a
        /// screen deciding to say something different about the same kind.
        case downloadCancelled
        case downloadSourceUnavailable
        case requestFailed
        /// The conversation (or a single message) is longer than the model's
        /// context window. Distinct from ``requestFailed`` because the remedy
        /// is specific and a plain "Try again" is actively wrong — resending
        /// the same too-long prompt fails identically. The engine emits
        /// ``context_length_exceeded``; the copy names the real fixes.
        case promptTooLong
        /// The requested model isn't available on the server. Engine code
        /// ``model_not_found``. Distinct from ``modelLoadFailed`` (a model that
        /// exists but couldn't start): here the name simply isn't served, so
        /// the destination is Model Management, not a file check.
        case modelUnavailable
        /// The primary model was replaced under a running request, so the
        /// engine cancelled it (code ``model_replacement``). Not a fault — a
        /// ``.notice`` — so it must not wear the alarming failure lane; the
        /// only recovery is to ask again against the new model.
        case requestSuperseded

        /// Forward-tolerant decode, matching ``ChatMessage/Role`` and
        /// ``ChatMessage/Status``: a raw value this build doesn't know (a
        /// transcript written by a NEWER build, then a downgrade) degrades to
        /// the generic ``.toolFailed`` rather than throwing. A throw here is
        /// not local — ``ConversationStore/load`` treats one undecodable
        /// message as a corrupt library and sides the whole history file, so
        /// the user's sidebar reads as wiped.
        init(from decoder: Decoder) throws {
            let raw = try decoder.singleValueContainer().decode(String.self)
            self = Kind(rawValue: raw) ?? .toolFailed
        }

        /// The nearest kind that EVERY shipped build already knows, used when
        /// writing to a key older builds decode strictly. Cases present since
        /// the first release map to themselves; anything added later names its
        /// closest ancestor here. See ``ChatMessage/encode(to:)``.
        var legacyPersistedKind: Kind {
            switch self {
            case .userDeclined:
                return .toolFailed
            // Added in #1580, after the same release ``userDeclined`` post-dates,
            // so it carries the same hazard: an older build decoding this raw
            // value throws, and one throw sides the ENTIRE history file.
            // ``webSearchUnavailable`` is the honest ancestor — it is the kind
            // this very condition used to land on, so an older build shows the
            // copy it always showed for a throttled search.
            case .webSearchRateLimited, .webSearchKeyRejected,
                 .webSearchKeyQuotaExceeded, .webSearchKeyRateLimited:
                return .webSearchUnavailable
            case .browsePageTooLarge:
                return .toolFailed
            // Added after the same release, and carrying the same hazard. A
            // download outcome is never written into a transcript today, but
            // the ancestor is named anyway so that stays true by construction
            // rather than by nobody having tried it.
            case .downloadCancelled:
                return .downloadFailed
            // Added after the same release, so each carries the downgrade
            // hazard: ``.requestFailed`` is the honest generic ancestor an
            // older build already knows — a chat that stopped short reads as
            // "couldn't finish that request", which is true for all three.
            case .promptTooLong, .modelUnavailable, .requestSuperseded:
                return .requestFailed
            case .modelOutOfMemory, .modelLoadFailed, .engineNotRunning,
                 .webSearchOffline, .webSearchUnavailable,
                 .commandPermissionDenied, .commandFailed,
                 .fileNotFound, .filePermissionDenied, .toolFailed,
                 .downloadFailed, .downloadSourceUnavailable, .requestFailed:
                return self
            }
        }
    }

    /// How a diagnosis should be PRESENTED, independent of what it says.
    ///
    /// Something the app (or the network, or the model) got wrong is an
    /// ``error``: red, alarming, worth interrupting for. Something the USER
    /// chose is a ``notice``: it explains why nothing came back and then gets
    /// out of the way. Painting a deliberate "Don't allow" red tells the user
    /// their own decision was a malfunction, and — because the copy on that
    /// lane is written for faults — blames their input for it.
    enum Severity: String, Equatable, Hashable, Sendable {
        case error
        case notice
    }

    /// A recovery affordance a failure card may offer. Every case must be
    /// something a view can actually carry out in THIS app — an action with no
    /// destination is the defect this type exists to prevent, and it is worse
    /// than no button because it fires at the moment the user has already been
    /// let down once.
    ///
    /// ``CaseIterable`` on purpose: it lets the routing table
    /// (``SettingsRouter/settingsCategory(for:)``) be pinned exhaustively by a
    /// test, so a new case cannot be added without a stated destination.
    ///
    /// Actions remain for active model, download, and chat failures.
    enum Action: String, CaseIterable, Equatable, Sendable {
        case retry
        case restart
        case openModelManagement
        case switchDownloadSource
        var title: String {
            switch self {
            case .retry: return "Retry"
            case .restart: return "Restart"
            case .openModelManagement: return "Open Model Management"
            case .switchDownloadSource: return "Switch source"
            }
        }

        var systemImage: String {
            switch self {
            case .retry, .restart: return "arrow.clockwise"
            case .openModelManagement: return "square.stack.3d.up"
            case .switchDownloadSource: return "arrow.triangle.2.circlepath"
            }
        }
    }

    let kind: Kind
    let message: String
    let action: Action?

    var severity: Severity { kind.severity }


}

extension FailureDiagnosis.Kind {
    /// Deliberately an exhaustive switch with no `default`: a new kind must
    /// state whether it is a fault the app is reporting or an outcome the user
    /// chose, rather than inheriting the alarming lane by omission.
    var severity: FailureDiagnosis.Severity {
        switch self {
        // Both of these are the user's own answer, not a malfunction: one
        // declines a permission prompt, the other stops a transfer already
        // running. Painting either red tells somebody their own decision
        // broke something.
        // A model swap under a running request cancelled it: the engine did
        // exactly what was asked (load the new model) and nothing malfunctioned,
        // so this is a calm notice, not a red fault. The recovery is simply to
        // ask again against the model that is now loaded.
        case .userDeclined, .downloadCancelled, .requestSuperseded:
            return .notice
        // A throttled backend is something that went wrong out in the world,
        // not something the user picked — it stays on the error lane, and its
        // copy and deep-link do the recovering.
        case .modelOutOfMemory, .modelLoadFailed, .engineNotRunning,
             .webSearchOffline, .webSearchUnavailable, .webSearchRateLimited,
             .webSearchKeyRejected, .webSearchKeyQuotaExceeded,
             .webSearchKeyRateLimited,
             .browsePageTooLarge,
             .commandPermissionDenied, .commandFailed,
             .fileNotFound, .filePermissionDenied, .toolFailed,
             .downloadFailed, .downloadSourceUnavailable, .requestFailed,
             .promptTooLong, .modelUnavailable:
            return .error
        }
    }
}

/// Rule-based classification for common failures. Matching deliberately uses
/// raw details only as input; none of those details are returned for display.
enum FailureDiagnoser {
    nonisolated static func diagnosis(
        for kind: FailureDiagnosis.Kind,
        modelAlias: String? = nil
    ) -> FailureDiagnosis {
        let message: String
        let action: FailureDiagnosis.Action?

        switch kind {
        case .modelOutOfMemory:
            if let modelAlias,
               ModelSizing.estimate(alias: modelAlias).paramsBillions != nil {
                let required = ModelSizing.estimate(alias: modelAlias).totalGB
                message = "This model needs about \(formatGB(required)) GB free. Free up memory or choose a smaller model."
            } else {
                message = "This model needs more free memory. Free up memory or choose a smaller model."
            }
            action = .openModelManagement
        case .modelLoadFailed:
            message = "This model couldn't load. Check the model files or choose another model."
            action = .openModelManagement
        case .engineNotRunning:
            message = "The local engine isn't running. Restart it, then try again."
            action = .restart
        case .webSearchOffline:
            message = "A web search in this older conversation could not connect."
            action = .retry
        case .webSearchUnavailable:
            message = "A web search in this older conversation could not finish."
            action = .retry
        case .webSearchKeyRejected:
            message = "A web search in this older conversation failed because its API key was rejected."
            action = nil
        case .webSearchKeyQuotaExceeded:
            message = "A web search in this older conversation exceeded its API key quota."
            action = nil
        case .webSearchKeyRateLimited:
            message = "A web search in this older conversation was rate-limited."
            action = nil
        case .webSearchRateLimited:
            // Deliberately NOT "check its settings": everything in Settings is
            // already correct when this fires. DuckDuckGo rate-limits the free
            // endpoint per IP within a handful of searches, so the honest
            // remedy is a different backend, and the message has to say which
            // ones and where. Kept to one sentence of situation + one of
            // remedy so it still reads inside the tool card. Not Brave any
            // more (#2043): Brave requires a card on file now, so pitching it
            // as the free fix would steer the user into surprise billing.
            message = "A web search in this older conversation was rate-limited."
            action = nil
        case .browsePageTooLarge:
            message = "This page is too large for Rapid to read at once. Search it or open a smaller page instead."
            action = nil
        case .commandPermissionDenied:
            // No action, and no "allow that folder, then try again" — this app
            // ships no shell tool and no place to grant a folder, so the old
            // copy asked for a control that does not exist and the old button
            // led nowhere. See ``FailureDiagnosis/Action``. What is left states
            // the outcome and stops.
            message = "The command tried to change a protected location, so it was blocked."
            action = nil
        case .commandFailed:
            message = "The command didn't finish successfully. Check the command, then try again."
            action = .retry
        case .fileNotFound:
            message = "That file isn't there. Check its name or location, then try again."
            action = .retry
        case .filePermissionDenied:
            // Same reasoning as ``commandPermissionDenied``: no file tools ship
            // here, so there is no access to grant and nothing to open.
            message = "Rapid doesn't have access to that file."
            action = nil
        case .toolFailed:
            // "Check its input" pointed the user at the one thing they do not
            // control: the arguments are written by the model, and a user
            // reading "check its input" over an invented `{"url": ""}` has
            // nothing to check (0.14.1 mini dogfood). State whose step failed
            // and leave the recovery — asking again — on the button.
            message = "A tool step didn't go through. The model can usually recover if you ask again."
            action = .retry
        case .userDeclined:
            // Nothing went wrong, so the copy states the outcome and stops.
            // No action button: the app has nothing to fix and nothing to
            // recover, and a prominent "Retry" would offer to re-run the exact
            // thing the user just refused — one stray click away from the
            // permission prompt existing for nothing. A user who declined by
            // mistake simply asks again, and the transcript's own per-message
            // Retry is still right there for a mis-click.
            // "nothing to show" rather than "nothing happened": a redirect can
            // be declined after the approved page has already been fetched, so
            // the honest claim is about the result, not the whole exchange.
            message = "You didn't allow this, so there's nothing to show. Ask again if you change your mind."
            action = nil
        case .downloadFailed:
            message = "The model download didn't finish. Check your connection, then try again."
            action = .retry
        case .downloadCancelled:
            // States what the user did, what it left behind, and what the one
            // button will do — and stops. No connection advice (nothing was
            // wrong with the connection), and deliberately silent about
            // whether a fresh pull reuses bytes already on disk: that is a
            // property of the downloader, not a promise this app is in a
            // position to make. See Paper 05.1 state 10 — "No Pause and no
            // resume."
            message = "You stopped this download, so the model isn't installed. Download it again, or choose a different model."
            action = .retry
        case .downloadSourceUnavailable:
            message = "The current download source couldn't be reached. Switch source and try again."
            action = .switchDownloadSource
        case .requestFailed:
            message = "Rapid couldn't finish that request. Try again, or restart the model."
            action = .retry
        case .promptTooLong:
            // No plain "Try again": resending the same over-long prompt fails
            // identically. Name the two fixes the user owns (shorten / new
            // chat) and offer the one this app can carry out — pick a
            // larger-context model — as the button.
            message = "This conversation is longer than the model's context window. Shorten your message or start a new chat, or switch to a model with a larger context window."
            action = .openModelManagement
        case .modelUnavailable:
            message = "That model isn't available on the server. Open Model Management to choose or start one."
            action = .openModelManagement
        case .requestSuperseded:
            // A ``.notice`` (see ``severity``): the model was switched on
            // purpose, so state what happened plainly and offer to ask again
            // against the model that is now loaded.
            message = "The model was switched, so this request stopped. Ask again to continue."
            action = .retry
        }
        return FailureDiagnosis(kind: kind, message: message, action: action)
    }

    nonisolated static func downloadFailureKind(
        raw: String,
        usingMirror: Bool
    ) -> FailureDiagnosis.Kind {
        guard usingMirror else { return .downloadFailed }
        let value = raw.lowercased()
        let mirrorSignals = offlineSignals + [
            "mirror", "models.rapidmlx.com", "cloudflare", "bad gateway",
            "service unavailable", "gateway timeout", "status 502", "status 503", "status 504",
        ]
        return containsAny(value, mirrorSignals) ? .downloadSourceUnavailable : .downloadFailed
    }

    nonisolated static func modelLoadFailureKind(raw: String) -> FailureDiagnosis.Kind {
        let value = raw.lowercased()
        return containsAny(value, memorySignals) ? .modelOutOfMemory : .modelLoadFailed
    }

    nonisolated static func engineFailureKind(raw: String) -> FailureDiagnosis.Kind {
        let value = raw.lowercased()
        if containsAny(value, memorySignals) { return .modelOutOfMemory }
        if containsAny(value, modelLoadSignals) { return .modelLoadFailed }
        return .engineNotRunning
    }

    /// Stable server error-code -> diagnosis kind. The engine emits a
    /// machine-readable ``error.code`` in its OpenAI-shaped envelope
    /// (rapid-mlx #3564) precisely so the GUI can classify a failure by
    /// CATEGORY without substring-matching a message the engine has already
    /// sanitised (a generation-time OOM reaches the client as a bare
    /// ``"Internal server error"`` -- no keyword can recover it). Unknown or
    /// absent codes return ``nil`` so the caller falls back to the keyword +
    /// status heuristics (older engines, or codes this build doesn't know yet).
    nonisolated static func kind(forEngineCode code: String?) -> FailureDiagnosis.Kind? {
        switch code {
        case "insufficient_memory", "model_out_of_memory":
            return .modelOutOfMemory
        case "model_load_failed":
            return .modelLoadFailed
        case "engine_aborted":
            // A genuine transient engine abort: retrying is the right recovery,
            // which is exactly what ``.requestFailed`` offers. OOM is split out
            // above so it gets the memory-specific card instead of a retry that
            // would fail identically.
            return .requestFailed
        case "context_length_exceeded":
            // The prompt/conversation is past the window. A retry of the same
            // input fails identically, so this must NOT collapse to the
            // generic retry card.
            return .promptTooLong
        case "model_not_found":
            return .modelUnavailable
        case "model_replacement":
            // Cooperative cancel from a model swap — a calm notice, not a
            // fault. Kept off ``.requestFailed`` so it isn't painted red.
            return .requestSuperseded
        default:
            return nil
        }
    }

    /// Decode the stable ``error.code`` from a rapid-mlx OpenAI-shaped error
    /// body, or ``nil`` when the body isn't that shape. Reuses
    /// ``Wire.ErrorEnvelope`` -- the same decoder ``ChatStreamClient`` uses for
    /// attachment-rejection codes -- so the app has ONE on-the-wire error
    /// schema, not two.
    nonisolated static func engineErrorCode(fromBody body: String) -> String? {
        guard let data = body.data(using: .utf8),
              let envelope = try? JSONDecoder().decode(
                Wire.ErrorEnvelope.self, from: data
              ),
              let code = envelope.error.code?.trimmingCharacters(
                in: .whitespacesAndNewlines
              ),
              !code.isEmpty
        else { return nil }
        return code
    }

    nonisolated static func chatFailureKind(raw: String) -> FailureDiagnosis.Kind {
        // Structured code first -- it survives the engine's message
        // sanitisation (#3564), unlike the keyword scan below.
        if let mapped = kind(forEngineCode: engineErrorCode(fromBody: raw)) {
            return mapped
        }
        let value = raw.lowercased()
        if containsAny(value, memorySignals) {
            return .modelOutOfMemory
        }
        if containsAny(value, [
            "can't reach the model", "couldn't reach the model", "disconnected",
            "connection refused", "local engine isn't running",
        ]) {
            return .engineNotRunning
        }
        return .requestFailed
    }

    nonisolated static func chatFailureKind(error: Error) -> FailureDiagnosis.Kind {
        if let chat = error as? ChatStreamError {
            switch chat {
            case .streamTruncated:
                return .engineNotRunning
            case .httpStatus(_, let body), .transport(let body):
                // Prefer the engine's stable, machine-readable error.code
                // (#3564): it survives the engine's message sanitisation, so
                // it is the only reliable category signal on a sanitised 5xx
                // body (a generation-time OOM arrives as "Internal server
                // error" with no keyword to match).
                if let mapped = kind(forEngineCode: engineErrorCode(fromBody: body)) {
                    return mapped
                }
                if modelLoadFailureKind(raw: body) == .modelOutOfMemory {
                    return .modelOutOfMemory
                }
                return .requestFailed
            }
        }
        let ns = error as NSError
        if ns.domain == NSURLErrorDomain {
            switch ns.code {
            case NSURLErrorCannotConnectToHost, NSURLErrorCannotFindHost,
                 NSURLErrorNetworkConnectionLost:
                return .engineNotRunning
            default:
                return .requestFailed
            }
        }
        return .requestFailed
    }

    nonisolated private static func containsAny(_ value: String, _ signals: [String]) -> Bool {
        signals.contains(where: value.contains)
    }

    nonisolated private static let offlineSignals = [
        "not connected to the internet", "network is unreachable", "network connection was lost",
        "could not resolve host", "cannot find host", "cannot connect", "connection refused",
        "connection reset", "dns", "offline", "timed out", "timeout",
    ]

    nonisolated private static let memorySignals = [
        "out of memory", "insufficient memory", "memory pressure", "metal-cap",
        "gpu_memory_utilization", "projected kv", "metal active",
        // User-facing phrasings the chat lane historically matched (#3564:
        // folded into the shared list so every diagnosis path — chat, load,
        // and 5xx body — recognises the same memory signals, not two divergent
        // sets).
        "more memory", "memory than your mac",
    ]

    nonisolated private static let modelLoadSignals = [
        "couldn't start the model", "could not start the model",
        "couldn't load the model", "could not load the model", "failed to load the model",
        "model name isn't valid", "model files",
    ]

    nonisolated private static func formatGB(_ value: Double) -> String {
        let rounded = ceil(value * 10) / 10
        if rounded.rounded() == rounded { return String(Int(rounded)) }
        return String(format: "%.1f", rounded)
    }
}
