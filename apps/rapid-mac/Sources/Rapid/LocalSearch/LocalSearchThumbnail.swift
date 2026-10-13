import AppKit
import AVFoundation
import QuickLookThumbnailing
import SwiftUI

/// Frame decoding and Quick Look rendering run off the main thread. Only the
/// completed image reaches the UI; the bounded cache is never written to disk.
final class LocalSearchThumbnailCache: @unchecked Sendable {
    static let shared = LocalSearchThumbnailCache()
    private let images = NSCache<NSString, NSImage>()
    typealias FrameLoader = @Sendable (String, Double, CGSize) -> CGImage?
    typealias QuickLookLoader = @Sendable (String, CGSize, @escaping @Sendable (CGImage?) -> Void) -> Void
    private let frameLoader: FrameLoader
    private let quickLookLoader: QuickLookLoader
    init(frameLoader: @escaping FrameLoader = LocalSearchThumbnailCache.videoFrame,
         quickLookLoader: @escaping QuickLookLoader = LocalSearchThumbnailCache.quickLook) {
        self.frameLoader = frameLoader
        self.quickLookLoader = quickLookLoader
        images.countLimit = 160
        images.totalCostLimit = 32 * 1024 * 1024
    }
    func thumbnail(path: String, modifiedAt: Double, size: Int, timestamp: Double? = nil,
                   pixelSize: CGSize = CGSize(width: 320, height: 240)) async -> NSImage? {
        let key = "\(path):\(modifiedAt):\(size):\(timestamp.map { String($0) } ?? "still"):\(pixelSize.width)x\(pixelSize.height)"
        if let image = images.object(forKey: key as NSString) { return image }
        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                if let timestamp, timestamp.isFinite, timestamp >= 0 {
                    if let frame = self.frameLoader(path, timestamp, pixelSize) {
                        let image = NSImage(cgImage: frame, size: CGSize(width: frame.width, height: frame.height))
                        self.images.setObject(image, forKey: key as NSString, cost: frame.bytesPerRow * frame.height)
                        continuation.resume(returning: image)
                        return
                    }
                }
                self.quickLookLoader(path, pixelSize) { frame in
                    guard let frame else { continuation.resume(returning: nil); return }
                    let image = NSImage(cgImage: frame, size: CGSize(width: frame.width, height: frame.height))
                    self.images.setObject(image, forKey: key as NSString, cost: frame.bytesPerRow * frame.height)
                    continuation.resume(returning: image)
                }
            }
        }
    }
    private static func videoFrame(path: String, timestamp: Double, size: CGSize) -> CGImage? {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: URL(fileURLWithPath: path)))
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = size
        // Default tolerance can choose a keyframe from another scene.
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        return try? generator.copyCGImage(at: CMTime(seconds: timestamp, preferredTimescale: 600), actualTime: nil)
    }
    private static func quickLook(path: String, size: CGSize, completion: @escaping @Sendable (CGImage?) -> Void) {
        let request = QLThumbnailGenerator.Request(fileAt: URL(fileURLWithPath: path),
            size: size, scale: 1, representationTypes: .thumbnail)
        QLThumbnailGenerator.shared.generateBestRepresentation(for: request) { representation, _ in
            completion(representation?.cgImage)
        }
    }

}

struct LocalSearchThumbnail: View {
    let result: LocalSearchResult
    @State private var image: NSImage?
    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 6).fill(RapidTheme.hoverFill)
            if let image {
                Image(nsImage: image).resizable().scaledToFit()
            } else {
                Image(systemName: result.kind == .video ? "video" : "photo")
                    .font(.title2).foregroundStyle(.secondary)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .accessibilityHidden(true)
        .task(id: "\(result.path):\(result.modifiedAt):\(result.size):\(result.timestamp ?? -1)") {
            image = nil
            let thumbnail = await LocalSearchThumbnailCache.shared.thumbnail(
                path: result.path, modifiedAt: result.modifiedAt, size: result.size,
                timestamp: result.kind == .video ? result.timestamp : nil)
            guard !Task.isCancelled else { return }
            image = thumbnail
        }
    }
}
