import Foundation

enum GoosePermissionAnswer: Equatable {
    case allow
    case decline
    case ask
}

struct GoosePermissionDecision: Equatable {
    let answer: GoosePermissionAnswer
    let cardState: String
}

enum GoosePermissionPolicy {
    static let yoloKey = "Rapid.GooseChat.yoloDefault"
    static let allowedKey = "Rapid.GooseChat.alwaysAllowed"

    static func decide(yolo: Bool, alwaysAllowed: Set<String>, tool: String,
                       userDecision: GoosePermissionAnswer? = nil) -> GoosePermissionDecision {
        if yolo { return .init(answer: .allow, cardState: "auto-allowed") }
        if alwaysAllowed.contains(tool) { return .init(answer: .allow, cardState: "auto-allowed") }
        switch userDecision {
        case .allow: return .init(answer: .allow, cardState: "allowed")
        case .decline: return .init(answer: .decline, cardState: "declined")
        case .ask, .none: return .init(answer: .ask, cardState: "permission needed")
        }
    }

    static func decide(tool: String, defaults: UserDefaults = .standard) -> GoosePermissionDecision {
        decide(yolo: defaults.bool(forKey: yoloKey),
               alwaysAllowed: Set(defaults.stringArray(forKey: allowedKey) ?? []), tool: tool)
    }

    static func alwaysAllow(_ tool: String, defaults: UserDefaults = .standard) {
        var names = Set(defaults.stringArray(forKey: allowedKey) ?? [])
        names.insert(tool)
        defaults.set(names.sorted(), forKey: allowedKey)
    }
}

enum GooseChatEvent {
    case session(String)
    case text(String)
    case thinking(String)
    case tool(id: String, name: String, input: String, status: String, output: String)
    case permission(WorkPermission, String, String)
    case error(String)
    case cancelled
    case throughput(Double)
}

struct GooseToolCard: Codable, Equatable, Hashable {
    var id: String
    var name: String
    var input: String
    var output: String = ""
    var status: String = "running"
    var permissionState: String? = nil
}

/// Ordered pieces of one ACP assistant turn. A tool updates its existing
/// piece, while text after the tool starts a new piece in the same message.
struct GooseTurnPart: Codable, Equatable, Hashable, Identifiable {
    let id: UUID
    var text: String?
    var tool: GooseToolCard?

    init(id: UUID = UUID(), text: String? = nil, tool: GooseToolCard? = nil) {
        self.id = id
        self.text = text
        self.tool = tool
    }
}
