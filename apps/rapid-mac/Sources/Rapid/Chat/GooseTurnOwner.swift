import Foundation

/// The identity of one helper turn. A late await or ACP event may only act on
/// the generation and session that created it.
@MainActor
final class GooseTurnOwner {
    private(set) var generation = 0
    private(set) var sessionID: String?
    private(set) var folder: URL?
    private var outstandingPermissions: Set<String> = []
    private var seenPermissions: Set<String> = []

    func reset() {
        generation &+= 1
        sessionID = nil
        folder = nil
        outstandingPermissions.removeAll()
        seenPermissions.removeAll()
    }

    func bind(sessionID: String, folder: URL) {
        self.sessionID = sessionID
        self.folder = folder
        outstandingPermissions.removeAll()
        seenPermissions.removeAll()
    }

    func accepts(_ sessionID: String?, generation: Int) -> Bool {
        self.generation == generation && (sessionID == nil || self.sessionID == sessionID)
    }

    private func key(_ request: WorkPermission) -> String {
        "\(request.sessionID):\(request.id)"
    }

    func register(_ request: WorkPermission) -> Bool {
        guard request.sessionID == sessionID else { return false }
        guard seenPermissions.insert(key(request)).inserted else { return false }
        outstandingPermissions.insert(key(request))
        return true
    }

    func consume(_ request: WorkPermission) -> Bool {
        guard request.sessionID == sessionID else { return false }
        return outstandingPermissions.remove(key(request)) != nil
    }
}

/// Small seam for wire fixtures and the live URLSession implementation.
@MainActor struct ACPTransport {
    var send: (URLRequest) async throws -> (Data, URLResponse)
    static let live = ACPTransport { try await URLSession.shared.data(for: $0) }
}
