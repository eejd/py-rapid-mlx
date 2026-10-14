import AppKit
import Darwin
import Foundation
import Observation

struct WorkStep: Identifiable {
    let id: String
    var command: String
    var status: String
    var output: String = ""
}

struct WorkPermission: Identifiable {
    let id: String
    let sessionID: String
    let command: String
    let options: [[String: Any]]
}

@MainActor @Observable
final class WorkAgent {
    static let shared = WorkAgent()
    private(set) var text = ""
    private(set) var steps: [WorkStep] = []
    private(set) var isRunning = false
    private(set) var isBusy = false
    var error: String?
    var permission: WorkPermission?
    var syntheticTestAutoApproval = false
    private var process: Process?
    private var port = 0
    private var secret = ""
    private var connection: String?
    private var sessionID: String?
    private var nextID = 1
    private var waiters: [String: CheckedContinuation<Data, Error>] = [:]
    private var streams: [Task<Void, Never>] = []
    private var folder: URL?
    private var serverIdentity: String?
    private var sandboxProfileURL: URL?

    private init() {}

    func prepare(folder: URL, rapidPort: Int, model: String, bearer: String) async throws {
        let identity = "\(rapidPort):\(model):\(bearer)"
        if isRunning, sessionID != nil, self.folder == folder, serverIdentity == identity { return }
        stop()
        let folder = folder.resolvingSymlinksInPath()
        self.folder = folder
        serverIdentity = identity
        error = nil
        text = ""
        steps = []
        let fileManager = FileManager.default
        let resources = Bundle.main.resourceURL!
        let goose = resources.appendingPathComponent("Helpers/goose")
        guard fileManager.isExecutableFile(atPath: goose.path) else {
            throw WorkError.message("Work helper is missing from this build.")
        }
        let python = resources.appendingPathComponent("rapid-mlx/python/bin")
        let support = fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/Rapid/agent", isDirectory: true)
        let temporary = support.appendingPathComponent("tmp", isDirectory: true)
        let providerDirectory = support.appendingPathComponent("config/custom_providers", isDirectory: true)
        try fileManager.createDirectory(at: support, withIntermediateDirectories: true,
                                        attributes: [.posixPermissions: 0o700])
        try fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: support.path)
        for directory in [temporary, providerDirectory] {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true,
                                            attributes: [.posixPermissions: 0o700])
            try fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        }
        let provider: [String: Any] = [
            "name": "custom_rapid_local", "engine": "openai", "display_name": "Rapid Local",
            "api_key_env": "RAPID_LOCAL_WORK_KEY",
            "base_url": "http://127.0.0.1:\(rapidPort)/v1/chat/completions",
            "models": [["name": model, "context_limit": 16384]],
            "supports_streaming": true, "requires_auth": true,
        ]
        let providerURL = providerDirectory.appendingPathComponent("custom_rapid_local.json")
        try JSONSerialization.data(withJSONObject: provider).write(to: providerURL, options: .atomic)
        let skillDirectory = support.appendingPathComponent(".agents/skills/local-documents", isDirectory: true)
        try fileManager.createDirectory(at: skillDirectory, withIntermediateDirectories: true)
        let skillSource = resources.appendingPathComponent("Helpers/local-documents/SKILL.md")
        try? fileManager.removeItem(at: skillDirectory.appendingPathComponent("SKILL.md"))
        try fileManager.copyItem(at: skillSource, to: skillDirectory.appendingPathComponent("SKILL.md"))
        port = try Self.freePort()
        secret = UUID().uuidString + UUID().uuidString
        let profile = Self.profile(folder: folder, support: support, temporary: temporary,
                                   resources: resources, rapidPort: rapidPort, goosePort: port)
        let profileURL = temporary.appendingPathComponent("work-\(UUID().uuidString).sb")
        try profile.write(to: profileURL, atomically: true, encoding: .utf8)
        sandboxProfileURL = profileURL
        let child = Process()
        child.executableURL = URL(fileURLWithPath: "/usr/bin/sandbox-exec")
        child.arguments = ["-f", profileURL.path, goose.path, "serve", "--host", "127.0.0.1",
                           "--port", String(port), "--platform", "desktop", "--with-builtin", "developer"]
        child.currentDirectoryURL = folder
        // Start from an allowlist: the shell tool must not inherit the app's
        // unrelated credentials or proxy configuration.
        var env: [String: String] = [
            "HOME": support.path,
            "USER": NSUserName(),
            "LOGNAME": NSUserName(),
            "LANG": "en_US.UTF-8",
            "NO_PROXY": "127.0.0.1,localhost",
        ]
        env["GOOSE_PATH_ROOT"] = support.path
        env["GOOSE_SERVER__SECRET_KEY"] = secret
        env["GOOSE_TELEMETRY_OFF"] = "1"
        env["GOOSE_TELEMETRY_ENABLED"] = "false"
        env["GOOSE_DISABLE_KEYRING"] = "1"
        env["GOOSE_PROVIDER"] = "custom_rapid_local"
        env["GOOSE_MODEL"] = model
        env["GOOSE_MODE"] = "approve"
        env["RAPID_LOCAL_WORK_KEY"] = bearer
        env["PATH"] = python.path + ":/usr/bin:/bin:/usr/sbin:/sbin"
        env["PYTHONPATH"] = resources.appendingPathComponent("rapid-mlx/site-packages").path
        env["PIP_NO_INDEX"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["TMPDIR"] = temporary.path
        child.environment = env
        child.standardOutput = FileHandle.nullDevice
        child.standardError = FileHandle.nullDevice
        try child.run()
        process = child
        var healthy = false
        for _ in 0..<60 {
            if !child.isRunning { break }
            if let url = URL(string: "http://127.0.0.1:\(port)/health"),
               let (_, response) = try? await URLSession.shared.data(from: url),
               (response as? HTTPURLResponse)?.statusCode == 200 {
                healthy = true
                break
            }
            try await Task.sleep(for: .milliseconds(500))
        }
        guard healthy else {
            stop()
            throw WorkError.message("The Work helper did not start under the privacy sandbox.")
        }
        isRunning = true
        let (initResult, response) = try await sendImmediate(method: "initialize", params: [
            "protocolVersion": 1,
            "clientCapabilities": ["fs": ["readTextFile": false, "writeTextFile": false]],
            "clientInfo": ["name": "Rapid Work", "version": "0.1"],
        ])
        guard initResult["error"] == nil,
              let id = response.value(forHTTPHeaderField: "Acp-Connection-Id") else {
            throw WorkError.message("The Work connection failed to initialize.")
        }
        connection = id
        openStream(session: nil)
        try await newSession()
    }

    func newSession() async throws {
        guard let folder else { return }
        if let old = sessionID {
            streams.last?.cancel()
            try? await notify(method: "session/cancel", params: ["sessionId": old], session: old)
        }
        text = ""
        steps = []
        permission = nil
        let result = try await call(method: "session/new", params: [
            "cwd": folder.path, "mcpServers": [],
            "_meta": ["client": "Rapid Work"],
        ])
        guard let sid = result["sessionId"] as? String else {
            throw WorkError.message("The Work session could not be created.")
        }
        sessionID = sid
        openStream(session: sid)
        try await Task.sleep(for: .milliseconds(300))
    }

    func prompt(_ message: String) async {
        guard let sessionID else { return }
        isBusy = true
        error = nil
        do {
            let instruction = "Use the local-documents skill. Stay inside the selected folder. " +
                "Read PDF with pypdf, CSV with csv.DictReader, XLSX with openpyxl. " +
                "Use Decimal for money, verify totals, format currency with thousands separators and two decimal places, and write results in the folder. " +
                "Do not install packages or use the network. User request: " + message
            _ = try await call(method: "session/prompt", params: [
                "sessionId": sessionID,
                "prompt": [["type": "text", "text": instruction]],
            ], session: sessionID, timeout: 900)
        } catch {
            self.error = error.localizedDescription
        }
        isBusy = false
    }

    func cancel() {
        guard let sessionID else { return }
        Task { try? await notify(method: "session/cancel", params: ["sessionId": sessionID], session: sessionID) }
        isBusy = false
    }

    func answerPermission(allow: Bool) {
        guard let permission else { return }
        let kind = allow ? "allow_once" : "reject_once"
        guard let option = permission.options.first(where: { $0["kind"] as? String == kind }),
              let optionID = option["optionId"] as? String else { return }
        self.permission = nil
        Task {
            try? await post(["jsonrpc": "2.0", "id": permission.id,
                             "result": ["outcome": ["outcome": "selected", "optionId": optionID]]],
                            session: permission.sessionID)
        }
    }

    func stop() {
        streams.forEach { $0.cancel() }
        streams = []
        if process?.isRunning == true { process?.terminate() }
        process = nil
        if let sandboxProfileURL { try? FileManager.default.removeItem(at: sandboxProfileURL) }
        sandboxProfileURL = nil
        isRunning = false
        isBusy = false
        connection = nil
        sessionID = nil
        secret = ""
        for (_, continuation) in waiters {
            continuation.resume(throwing: WorkError.message("Work stopped."))
        }
        waiters = [:]
    }

    private func openStream(session: String?) {
        let connection = self.connection
        let port = self.port
        let secret = self.secret
        streams.append(Task {
            guard let url = URL(string: "http://127.0.0.1:\(port)/acp") else { return }
            var request = URLRequest(url: url)
            request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
            request.setValue(secret, forHTTPHeaderField: "X-Secret-Key")
            request.setValue(connection, forHTTPHeaderField: "Acp-Connection-Id")
            request.setValue(session, forHTTPHeaderField: "Acp-Session-Id")
            do {
                let (bytes, _) = try await URLSession.shared.bytes(for: request)
                for try await line in bytes.lines {
                    if Task.isCancelled { return }
                    guard line.hasPrefix("data:"),
                          let data = String(line.dropFirst(5)).trimmingCharacters(in: .whitespaces).data(using: .utf8),
                          let message = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
                    handle(message)
                }
            } catch {
                if !Task.isCancelled { self.error = "Work stream ended: \(error.localizedDescription)" }
            }
        })
    }

    private func handle(_ message: [String: Any]) {
        if let id = message["id"], message["method"] == nil,
           let continuation = waiters.removeValue(forKey: String(describing: id)) {
            if let failure = message["error"] {
                continuation.resume(throwing: WorkError.message("ACP error: \(failure)"))
            } else {
                let result = message["result"] as? [String: Any] ?? [:]
                if let data = try? JSONSerialization.data(withJSONObject: result) {
                    continuation.resume(returning: data)
                } else {
                    continuation.resume(throwing: WorkError.message("Invalid Work response."))
                }
            }
            return
        }
        if message["method"] as? String == "session/request_permission",
           let params = message["params"] as? [String: Any],
           let id = message["id"],
           let sid = params["sessionId"] as? String,
           let tool = params["toolCall"] as? [String: Any] {
            let input = tool["rawInput"] as? [String: Any]
            permission = WorkPermission(id: String(describing: id), sessionID: sid,
                command: input?["command"] as? String ?? tool["title"] as? String ?? "Unknown command",
                options: params["options"] as? [[String: Any]] ?? [])
            if syntheticTestAutoApproval { answerPermission(allow: true) }
            return
        }
        guard message["method"] as? String == "session/update",
              let params = message["params"] as? [String: Any],
              let update = params["update"] as? [String: Any],
              let kind = update["sessionUpdate"] as? String else { return }
        switch kind {
        case "agent_message_chunk":
            if let content = update["content"] as? [String: Any],
               content["type"] as? String == "text", let chunk = content["text"] as? String {
                text += chunk
            }
        case "tool_call":
            let id = update["toolCallId"] as? String ?? UUID().uuidString
            let input = update["rawInput"] as? [String: Any]
            steps.append(WorkStep(id: id, command: input?["command"] as? String ?? update["title"] as? String ?? "Tool", status: "Running"))
        case "tool_call_update":
            guard let id = update["toolCallId"] as? String,
                  let index = steps.firstIndex(where: { $0.id == id }) else { return }
            steps[index].status = update["status"] as? String ?? "Updated"
            if let raw = update["rawOutput"] as? [String: Any] {
                steps[index].output = (raw["stdout"] as? String ?? "") + (raw["stderr"] as? String ?? "")
            }
        default: break
        }
    }

    private func call(method: String, params: [String: Any], session: String? = nil,
                      timeout: TimeInterval = 90) async throws -> [String: Any] {
        let id = nextID
        nextID += 1
        let data: Data = try await withCheckedThrowingContinuation { continuation in
            waiters[String(id)] = continuation
            Task {
                do {
                    try await post(["jsonrpc": "2.0", "id": id, "method": method, "params": params], session: session)
                } catch {
                    waiters.removeValue(forKey: String(id))?.resume(throwing: error)
                }
            }
            Task {
                try? await Task.sleep(for: .seconds(timeout))
                waiters.removeValue(forKey: String(id))?.resume(
                    throwing: WorkError.message("Work request timed out: \(method)"))
            }
        }
        guard let result = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw WorkError.message("Invalid Work response.")
        }
        return result
    }

    private func notify(method: String, params: [String: Any], session: String?) async throws {
        try await post(["jsonrpc": "2.0", "method": method, "params": params], session: session)
    }

    private func sendImmediate(method: String, params: [String: Any]) async throws -> ([String: Any], HTTPURLResponse) {
        let id = nextID
        nextID += 1
        let (data, response) = try await postData(["jsonrpc": "2.0", "id": id, "method": method, "params": params], session: nil)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw WorkError.message("Invalid Work initialization response.")
        }
        return (object, response)
    }

    private func post(_ body: [String: Any], session: String?) async throws {
        _ = try await postData(body, session: session)
    }

    private func postData(_ body: [String: Any], session: String?) async throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/acp")!)
        request.httpMethod = "POST"
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(secret, forHTTPHeaderField: "X-Secret-Key")
        request.setValue(connection, forHTTPHeaderField: "Acp-Connection-Id")
        request.setValue(session, forHTTPHeaderField: "Acp-Session-Id")
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let response = response as? HTTPURLResponse, response.statusCode < 400 else {
            throw WorkError.message("Work transport rejected a request.")
        }
        return (data, response)
    }

    private static func freePort() throws -> Int {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw WorkError.message("Could not allocate a local port.") }
        defer { close(fd) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_addr = in_addr(s_addr: in_addr_t(0x0100007f))
        let result = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        guard result == 0 else { throw WorkError.message("Could not bind a local port.") }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let found = withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &length) }
        }
        guard found == 0 else { throw WorkError.message("Could not read a local port.") }
        return Int(UInt16(bigEndian: address.sin_port))
    }

    private static func profile(folder: URL, support: URL, temporary: URL,
                                resources: URL, rapidPort: Int, goosePort: Int) -> String {
        func quoted(_ value: String) -> String {
            "\"" + value.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"") + "\""
        }
        // SQLite resolves every ancestor before opening sessions.db. Permit
        // metadata only on those ancestors; contents remain unreadable.
        var ancestors = Set<String>()
        for url in [folder, support, resources] {
            var parent = url.deletingLastPathComponent()
            while parent.path != "/" {
                ancestors.insert(parent.path)
                parent.deleteLastPathComponent()
            }
        }
        let ancestorRules = ancestors.sorted().map { "(literal \(quoted($0)))" }.joined(separator: " ")
        return """
        (version 1)
        (deny default)
        (allow process*)
        (allow mach-lookup)
        (allow sysctl-read)
        (allow file-read* (literal "/") (literal "/private")
            (subpath "/System") (subpath "/usr") (subpath "/bin")
            (subpath "/sbin") (subpath "/Library")
            (subpath "/private/etc") (subpath "/dev"))
        (allow file-read-metadata \(ancestorRules))
        (allow file-read* (subpath \(quoted(resources.path)))
            (subpath \(quoted(support.path))) (subpath \(quoted(folder.path))))
        (allow file-write* (literal "/dev/null") (subpath \(quoted(folder.path)))
            (subpath \(quoted(support.path))) (subpath \(quoted(temporary.path))))
        (allow network-inbound (local ip "localhost:*"))
        (allow network-outbound (remote ip "localhost:\(rapidPort)"))
        (allow network-outbound (remote ip "localhost:\(goosePort)"))
        """
    }
}

enum WorkError: LocalizedError {
    case message(String)
    var errorDescription: String? {
        if case .message(let value) = self { return value }
        return nil
    }
}
