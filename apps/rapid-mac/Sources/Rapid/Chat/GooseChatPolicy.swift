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
}

struct GooseToolCard: Codable, Equatable, Hashable {
    var id: String
    var name: String
    var input: String
    var output: String = ""
    var status: String = "running"
    var permissionState: String? = nil
    var startedAt: Date? = Date()
}

enum GooseToolPresentation {
    static func producedFile(_ card: GooseToolCard, folder: String) -> URL? {
        let lower = card.name.lowercased()
        guard (lower.contains("write") || lower.contains("edit") || lower.contains("patch")),
              !isRunning(card.status), !isFailure(card.status),
              let data = card.input.data(using: .utf8),
              let args = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let path = path(name: card.name, args: args) else { return nil }
        let root = URL(fileURLWithPath: folder, isDirectory: true).standardizedFileURL
        let file = (path.hasPrefix("/") ? URL(fileURLWithPath: path) : root.appendingPathComponent(path)).standardizedFileURL
        guard file.path.hasPrefix(root.path.hasSuffix("/") ? root.path : root.path + "/") else { return nil }
        return file
    }

    static func title(name: String, input: String, folder: String) -> String {
        let args = (input.data(using: .utf8).flatMap {
            try? JSONSerialization.jsonObject(with: $0) as? [String: Any]
        }) ?? [:]
        let lower = name.lowercased()
        let path = path(name: name, args: args)
        let relative: String? = path.map { value in
            let root = URL(fileURLWithPath: folder).standardizedFileURL.path
            let prefix = root.hasSuffix("/") ? root : root + "/"
            if value.hasPrefix(prefix) { return String(value.dropFirst(prefix.count)) }
            return value.hasPrefix("/") ? URL(fileURLWithPath: value).lastPathComponent : value
        }
        let subject = relative ?? String(localized: "file")
        if lower.contains("write") || lower.contains("create_file") { return String(localized: "Wrote \(subject)") }
        if lower.contains("edit") || lower.contains("patch") { return String(localized: "Edited \(subject)") }
        if lower.contains("shell") || lower.contains("command") || lower.contains("execute") || lower.contains("run") {
            let command = (args["command"] as? String ?? args["cmd"] as? String ?? name)
                .replacingOccurrences(of: "\n", with: " ")
                .replacingOccurrences(of: folder, with: ".")
            return String(localized: "Ran `\(String(command.prefix(90)))`")
        }
        if lower.contains("list") { return String(localized: "Listed files") }
        if lower.contains("open") || lower.contains("browse") { return String(localized: "Opened \(subject)") }
        if lower.contains("read") { return String(localized: "Read \(subject)") }
        let displayName = name.components(separatedBy: " · ").first ?? name
        return relative.map { "\(displayName) · \($0)" } ?? displayName
    }

    static func isRunning(_ status: String) -> Bool {
        ["running", "pending", "in_progress", "updated"].contains(status.lowercased())
    }

    static func isFailure(_ status: String) -> Bool {
        let value = status.lowercased()
        return value.contains("fail") || value.contains("error") || value.contains("cancel") || value.contains("declin")
    }

    static func outputPreview(_ output: String, expanded: Bool) -> String {
        let lines = output.components(separatedBy: .newlines)
        return expanded ? output : lines.prefix(40).joined(separator: "\n")
    }

    static func progress(card: GooseToolCard?, elapsed: Int, folder: String) -> String {
        let step: String
        if let card {
            let title = title(name: card.name, input: card.input, folder: folder)
            step = title.hasPrefix("Wrote ") ? title.replacingOccurrences(of: "Wrote ", with: "Writing ")
                : title.hasPrefix("Edited ") ? title.replacingOccurrences(of: "Edited ", with: "Editing ")
                : title.hasPrefix("Ran ") ? title.replacingOccurrences(of: "Ran ", with: "Running ")
                : title
        } else {
            step = String(localized: "Preparing response")
        }
        let size = card.flatMap { dataSize($0.input) }
        return "\(step)\(size.map { " · \($0)" } ?? "") · \(elapsed) s"
    }

    private static func dataSize(_ input: String) -> String? {
        guard let bytes = input.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: bytes) as? [String: Any],
              let content = ["content", "text", "file_content"].compactMap({ object[$0] as? String }).first
        else { return nil }
        return String(format: "%.1f KB", Double(content.utf8.count) / 1024)
    }

    private static func path(name: String, args: [String: Any]) -> String? {
        if let value = ["path", "file_path", "filename", "file", "target_path", "url"]
            .compactMap({ args[$0] as? String }).first { return value }
        // Goose's ACP title is often "write · /absolute/path" while rawInput
        // contains only the file content. Keep the path out of the collapsed row.
        if let tail = name.components(separatedBy: " · ").last, tail.hasPrefix("/") {
            return tail
        }
        return nil
    }
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
