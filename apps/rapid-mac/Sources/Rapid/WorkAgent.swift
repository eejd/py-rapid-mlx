import AppKit
import CryptoKit
import Darwin
import Foundation
import Observation

extension Notification.Name {
    static let newWorkChat = Notification.Name("Rapid.NewWorkChat")
}

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
    static var fullAccess: Bool { Bundle.main.bundleIdentifier == "com.rapidmlx.rapid.work-full" }
    private(set) var text = ""
    private(set) var steps: [WorkStep] = []
    private(set) var isRunning = false
    private(set) var isBusy = false
    private(set) var isPreparing = false
    private(set) var isOpeningSession = false
    private(set) var conversationRevision = 0
    var error: String?
    var permission: WorkPermission?
    var syntheticTestAutoApproval = false
    private var process: Process?
    private var proxyProcess: Process?
    private var port = 0
    private var proxyPort = 0
    private var secret = ""
    private var connection: String?
    private var sessionID: String?
    private var nextID = 1
    private var waiters: [String: CheckedContinuation<Data, Error>] = [:]
    private var streams: [Task<Void, Never>] = []
    private var folder: URL?
    private var serverIdentity: String?
    private var sandboxProfileURLs: [URL] = []
    private var pendingPermissions: [WorkPermission] = []
    private var generation = 0
    var lifecycleGeneration: Int { generation }

    private init() {}

    func beginNewChat() {
        conversationRevision += 1
        if isPreparing || isBusy || isOpeningSession { stop() }
        else if isRunning { isOpeningSession = true }
        text = ""
        steps = []
        error = nil
    }

    func prepare(folder: URL, rapidPort: Int, model: String, bearer: String) async throws {
        guard !isPreparing else { throw WorkError.message("Work is already starting.") }
        let requestedRevision = conversationRevision
        let folder = folder.resolvingSymlinksInPath()
        let identity = "\(rapidPort):\(model):\(bearer)"
        if isRunning, sessionID != nil, self.folder == folder, serverIdentity == identity { return }
        isPreparing = true
        defer { isPreparing = false }
        stop()
        let prepareGeneration = generation
        var completed = false
        defer { if !completed && generation == prepareGeneration { stop() } }
        self.folder = folder
        serverIdentity = identity
        error = nil
        text = ""
        steps = []
        let fileManager = FileManager.default
        let resources = Bundle.main.resourceURL!
        let goose = resources.appendingPathComponent("Helpers/goose")
        let proxy = resources.appendingPathComponent("Helpers/work-inference-proxy.py")
        guard fileManager.isExecutableFile(atPath: goose.path),
              fileManager.fileExists(atPath: proxy.path) else {
            throw WorkError.message("Work helper is missing from this build.")
        }
        let python = resources.appendingPathComponent("rapid-mlx/python/bin")
        let pythonExecutable = python.appendingPathComponent("python3")
        let rootSupport = fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent(Self.fullAccess ? "Library/Application Support/Rapid/agent-full" : "Library/Application Support/Rapid/agent", isDirectory: true)
        let folderHash = SHA256.hash(data: Data(folder.path.utf8))
            .map { String(format: "%02x", $0) }.joined()
        let support = rootSupport.appendingPathComponent("workspaces/\(folderHash)", isDirectory: true)
        let temporary = support.appendingPathComponent("tmp", isDirectory: true)
        let providerDirectory = support.appendingPathComponent("config/custom_providers", isDirectory: true)
        try fileManager.createDirectory(at: rootSupport, withIntermediateDirectories: true,
                                        attributes: [.posixPermissions: 0o700])
        try fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: rootSupport.path)
        try fileManager.createDirectory(at: support, withIntermediateDirectories: true,
                                        attributes: [.posixPermissions: 0o700])
        try fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: support.path)
        for directory in [temporary, providerDirectory] {
            try fileManager.createDirectory(at: directory, withIntermediateDirectories: true,
                                            attributes: [.posixPermissions: 0o700])
            try fileManager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        }
        port = try Self.freePort()
        repeat { proxyPort = try Self.freePort() } while proxyPort == port
        secret = UUID().uuidString + UUID().uuidString
        guard conversationRevision == requestedRevision else { throw WorkError.message("Work start cancelled.") }
        let provider: [String: Any] = [
            "name": "custom_rapid_local", "engine": "openai", "display_name": "Rapid Local",
            "api_key_env": "RAPID_LOCAL_WORK_KEY",
            "base_url": "http://127.0.0.1:\(proxyPort)/v1/chat/completions",
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
        let gooseProfileURL = temporary.appendingPathComponent("work-\(UUID().uuidString).sb")
        let proxyProfileURL = temporary.appendingPathComponent("proxy-\(UUID().uuidString).sb")
        if !Self.fullAccess {
            try Self.profile(folder: folder, support: support, temporary: temporary,
                             resources: resources, outboundPorts: [proxyPort, port])
                .write(to: gooseProfileURL, atomically: true, encoding: .utf8)
            sandboxProfileURLs.append(gooseProfileURL)
            try Self.profile(folder: folder, support: support, temporary: temporary,
                             resources: resources, outboundPorts: [rapidPort])
                .write(to: proxyProfileURL, atomically: true, encoding: .utf8)
            sandboxProfileURLs.append(proxyProfileURL)
        }
        // Start from an allowlist: the shell tool must not inherit the app's
        // unrelated credentials or proxy configuration.
        var env: [String: String] = [
            "HOME": Self.fullAccess ? fileManager.homeDirectoryForCurrentUser.path : support.path,
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
        env["GOOSE_MODE"] = Self.fullAccess ? "auto" : "approve"
        env["RAPID_LOCAL_WORK_KEY"] = bearer
        env["PATH"] = python.path + ":/usr/bin:/bin:/usr/sbin:/sbin"
        env["PYTHONPATH"] = resources.appendingPathComponent("rapid-mlx/site-packages").path
        if Self.fullAccess {
            let venv = rootSupport.appendingPathComponent("venv")
            let pipMarker = venv.appendingPathComponent(".rapid-pip-ready")
            if !fileManager.fileExists(atPath: pipMarker.path) {
                let setup = Process()
                setup.executableURL = pythonExecutable
                setup.arguments = ["-m", "venv", "--without-pip", venv.path]
                try setup.run()
                setup.waitUntilExit()
                guard setup.terminationStatus == 0 else { throw WorkError.message("Could not create Work Python venv.") }
                let bootstrap = Process()
                bootstrap.executableURL = venv.appendingPathComponent("bin/python3")
                bootstrap.arguments = ["-c", "import pathlib,sysconfig,zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sysconfig.get_path('purelib'))",
                                       resources.appendingPathComponent("Helpers/pip.whl").path]
                try bootstrap.run()
                bootstrap.waitUntilExit()
                guard bootstrap.terminationStatus == 0 else { throw WorkError.message("Could not install Work venv pip.") }
                let pipCommand = venv.appendingPathComponent("bin/pip")
                try "#!/bin/sh\nexec \"$(dirname \"$0\")/python3\" -m pip \"$@\"\n"
                    .write(to: pipCommand, atomically: true, encoding: .utf8)
                try fileManager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: pipCommand.path)
                try Data().write(to: pipMarker)
            }
            env["PATH"] = venv.appendingPathComponent("bin").path + ":" + env["PATH", default: ""]
            env["VIRTUAL_ENV"] = venv.path
        } else {
            env["PIP_NO_INDEX"] = "1"
        }
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["TMPDIR"] = temporary.path
        var proxyEnv = env
        proxyEnv.removeValue(forKey: "GOOSE_SERVER__SECRET_KEY")
        proxyEnv.removeValue(forKey: "RAPID_LOCAL_WORK_KEY")
        proxyEnv["WORK_PROXY_BEARER"] = bearer
        proxyProcess = try Self.launchHelper(
            profile: Self.fullAccess ? nil : proxyProfileURL, python: pythonExecutable,
            executable: pythonExecutable,
            arguments: ["-I", "-S", proxy.path, String(proxyPort), String(rapidPort), model]
                + (Self.fullAccess ? ["--full-access"] : []),
            directory: support, environment: proxyEnv)
        guard await Self.waitForHealth(process: proxyProcess!, port: proxyPort) else {
            throw WorkError.message("The local Work inference gate did not start.")
        }
        guard conversationRevision == requestedRevision else { throw WorkError.message("Work start cancelled.") }
        process = try Self.launchHelper(
            profile: Self.fullAccess ? nil : gooseProfileURL, python: pythonExecutable,
            executable: goose,
            arguments: ["serve", "--host", "127.0.0.1", "--port", String(port),
                        "--platform", "desktop", "--with-builtin", Self.fullAccess
                            ? "developer,computercontroller,memory,autovisualiser,tutorial"
                            : "developer"],
            directory: folder, environment: env)
        guard await Self.waitForHealth(process: process!, port: port) else {
            throw WorkError.message("The Work helper did not start under the privacy sandbox.")
        }
        guard conversationRevision == requestedRevision else { throw WorkError.message("Work start cancelled.") }
        isRunning = true
        let (initResult, response) = try await sendImmediate(method: "initialize", params: [
            "protocolVersion": 1,
            "clientCapabilities": ["fs": ["readTextFile": false, "writeTextFile": false]],
            "clientInfo": ["name": "Rapid Work", "version": "0.1"],
        ])
        guard conversationRevision == requestedRevision else { throw WorkError.message("Work start cancelled.") }
        guard initResult["error"] == nil,
              let id = response.value(forHTTPHeaderField: "Acp-Connection-Id") else {
            throw WorkError.message("The Work connection failed to initialize.")
        }
        connection = id
        openStream(session: nil)
        try await newSession()
        guard conversationRevision == requestedRevision else { throw WorkError.message("Work start cancelled.") }
        completed = true
    }

    func newSession() async throws {
        if !isRunning {
            isOpeningSession = false
            text = ""
            steps = []
            error = nil
            return
        }
        guard let folder else { return }
        guard !isBusy else { throw WorkError.message("Stop the current task before starting a new session.") }
        let sessionGeneration = generation
        isOpeningSession = true
        defer { if generation == sessionGeneration { isOpeningSession = false } }
        if let old = sessionID {
            try? await notify(method: "session/cancel", params: ["sessionId": old], session: old)
            guard generation == sessionGeneration else { throw WorkError.message("Work stopped.") }
            streams.last?.cancel()
        }
        text = ""
        steps = []
        error = nil
        permission = nil
        pendingPermissions = []
        let result = try await call(method: "session/new", params: [
            "cwd": folder.path, "mcpServers": [],
            "_meta": ["client": "Rapid Work"],
        ])
        guard generation == sessionGeneration else { throw WorkError.message("Work stopped.") }
        guard let sid = result["sessionId"] as? String else {
            throw WorkError.message("The Work session could not be created.")
        }
        sessionID = sid
        openStream(session: sid)
        try await Task.sleep(for: .milliseconds(300))
        guard generation == sessionGeneration else { throw WorkError.message("Work stopped.") }
    }

    func prompt(_ message: String) async {
        guard let sessionID else { return }
        let promptGeneration = generation
        isBusy = true
        error = nil
        do {
            let instruction = Self.fullAccess ? message : "Use the local-documents skill. Stay inside the selected folder. " +
                "Read PDF with pypdf, CSV with csv.DictReader, XLSX with openpyxl. " +
                "Use Decimal for money, verify totals, format currency with thousands separators and two decimal places, and write results in the folder. Put every total label and its amount on the same line. " +
                "Do not install packages, use the network, or start background or detached processes. User request: " + message
            _ = try await call(method: "session/prompt", params: [
                "sessionId": sessionID,
                "prompt": [["type": "text", "text": instruction]],
            ], session: sessionID, timeout: 900)
        } catch {
            if generation == promptGeneration { self.error = error.localizedDescription }
        }
        if generation == promptGeneration { isBusy = false }
    }

    func cancel() {
        guard let sessionID else { return }
        Task {
            try? await notify(method: "session/cancel", params: ["sessionId": sessionID], session: sessionID)
            stop()
        }
    }

    func answerPermission(_ request: WorkPermission, allow: Bool) {
        let kind = allow ? "allow_once" : "reject_once"
        guard let option = request.options.first(where: { $0["kind"] as? String == kind }),
              let optionID = option["optionId"] as? String else { return }
        if permission?.id == request.id {
            permission = pendingPermissions.isEmpty ? nil : pendingPermissions.removeFirst()
        }
        Task {
            try? await post(["jsonrpc": "2.0", "id": request.id,
                             "result": ["outcome": ["outcome": "selected", "optionId": optionID]]],
                            session: request.sessionID)
        }
    }

    func stop() {
        generation += 1
        streams.forEach { $0.cancel() }
        streams = []
        let children = [process, proxyProcess].compactMap { $0 }
        for child in children {
            let group = child.processIdentifier
            if group > 0 { _ = Darwin.kill(-group, SIGTERM) }
        }
        if !children.isEmpty { usleep(150_000) }
        for child in children {
            let group = child.processIdentifier
            if group > 0 { _ = Darwin.kill(-group, SIGKILL) }
        }
        process = nil
        proxyProcess = nil
        for profile in sandboxProfileURLs { try? FileManager.default.removeItem(at: profile) }
        sandboxProfileURLs = []
        isRunning = false
        isBusy = false
        isOpeningSession = false
        permission = nil
        pendingPermissions = []
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
            let request = WorkPermission(id: String(describing: id), sessionID: sid,
                command: input?["command"] as? String ?? tool["title"] as? String ?? "Unknown command",
                options: params["options"] as? [[String: Any]] ?? [])
            if syntheticTestAutoApproval || Self.fullAccess {
                answerPermission(request, allow: true)
            } else if permission == nil {
                permission = request
            } else {
                pendingPermissions.append(request)
            }
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
            if let content = update["content"] as? [[String: Any]] {
                let text = content.compactMap { item -> String? in
                    if let nested = item["content"] as? [String: Any],
                       nested["type"] as? String == "text" { return nested["text"] as? String }
                    if item["type"] as? String == "text" { return item["text"] as? String }
                    return nil
                }.joined(separator: "\n")
                if !text.isEmpty { steps[index].output = text }
            }
            if let meta = update["_meta"] as? [String: Any],
               let notification = meta["toolNotification"] as? [String: Any],
               let params = notification["params"] as? [String: Any],
               let chunks = params["chunks"] as? [[String: Any]] {
                steps[index].output += chunks.compactMap { $0["output"] as? String }.joined()
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

    private static func launchHelper(profile: URL?, python: URL, executable: URL,
                                        arguments: [String], directory: URL,
                                        environment: [String: String]) throws -> Process {
        let child = Process()
        child.executableURL = python
        child.arguments = ["-I", "-S", Bundle.main.resourceURL!.appendingPathComponent("Helpers/work-helper-watchdog.py").path,
                           String(getpid())] + (profile.map { ["/usr/bin/sandbox-exec", "-f", $0.path] } ?? [])
                           + [executable.path] + arguments
        child.currentDirectoryURL = directory
        child.environment = environment
        child.standardOutput = FileHandle.nullDevice
        if ProcessInfo.processInfo.arguments.contains("--work-mvp-synthetic-test") {
            let log = directory.appendingPathComponent("work-\(executable.lastPathComponent).log")
            FileManager.default.createFile(atPath: log.path, contents: nil)
            child.standardError = try FileHandle(forWritingTo: log)
        } else {
            child.standardError = FileHandle.nullDevice
        }
        try child.run()
        for _ in 0..<100 {
            if getpgid(child.processIdentifier) == child.processIdentifier { return child }
            if !child.isRunning { break }
            usleep(10_000)
        }
        if child.isRunning { child.terminate() }
        throw WorkError.message("Could not isolate the Work helper process group.")
    }

    private static func waitForHealth(process: Process, port: Int) async -> Bool {
        for _ in 0..<60 {
            if !process.isRunning { return false }
            if let url = URL(string: "http://127.0.0.1:\(port)/health"),
               let (_, response) = try? await URLSession.shared.data(from: url),
               (response as? HTTPURLResponse)?.statusCode == 200 {
                return true
            }
            try? await Task.sleep(for: .milliseconds(500))
        }
        return false
    }

    private static func profile(folder: URL, support: URL, temporary: URL,
                                resources: URL, outboundPorts: [Int]) -> String {
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
        let networkRules = outboundPorts.map {
            "(allow network-outbound (remote ip \"localhost:\($0)\"))"
        }.joined(separator: "\n")
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
        \(networkRules)
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
