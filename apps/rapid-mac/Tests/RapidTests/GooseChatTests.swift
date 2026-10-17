import Foundation
import CryptoKit
import SQLite3
import Testing
@testable import Rapid

@MainActor
@Suite("Goose Chat")
struct GooseChatTests {
    @Test("ACP session identity rejects foreign updates and merges out-of-order tool cards")
    func wireSessionAndTools() throws {
        let agent = WorkAgent()
        agent.acceptSessionForTesting("session-a", folder: URL(fileURLWithPath: "/tmp/a"))
        var events: [GooseChatEvent] = []
        agent.onChatEvent = { events.append($0) }
        func wire(_ json: String) throws { try agent.handleWire(Data(json.utf8)) }
        try wire(#"{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"session-b","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"foreign"}}}}"#)
        #expect(events.isEmpty)
        try wire(#"{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"session-a","update":{"sessionUpdate":"tool_call_update","toolCallId":"t1","status":"completed","rawOutput":{"stdout":"done"}}}}"#)
        try wire(#"{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"session-a","update":{"sessionUpdate":"tool_call","toolCallId":"t1","title":"Write","rawInput":{"path":"file.txt"}}}}"#)
        try wire(#"{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"session-a","update":{"sessionUpdate":"tool_call_update","toolCallId":"t1","content":[{"type":"text","text":"done"}]}}}"#)
        let cards = events.compactMap { event -> (String, String)? in
            if case let .tool(id, _, _, status, output) = event { return (status, output) }
            return nil
        }
        #expect(cards.count == 3)
        #expect(cards.allSatisfy { $0.0 == "completed" })
        #expect(cards.last?.1 == "done")
        agent.stop()
        try wire(#"{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"session-a","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"late"}}}}"#)
        #expect(events.count == 3)
        agent.acceptSessionForTesting("session-b", folder: URL(fileURLWithPath: "/tmp/b"))
        try wire(#"{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"session-a","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"old session"}}}}"#)
        #expect(events.count == 3)
    }

    @Test("ACP permission identity is consumed once and reset with the turn")
    func permissionOwner() {
        let owner = GooseTurnOwner()
        let folder = URL(fileURLWithPath: "/tmp/a")
        owner.bind(sessionID: "session-a", folder: folder)
        let request = WorkPermission(id: "42", sessionID: "session-a", toolCallID: "t1", command: "run", options: [])
        #expect(owner.register(request))
        #expect(!owner.register(request))
        #expect(owner.consume(request))
        #expect(!owner.consume(request))
        owner.reset()
        #expect(!owner.register(request))
        #expect(!owner.accepts("session-a", generation: 0))
        owner.bind(sessionID: "session-b", folder: folder)
        #expect(!owner.register(request))
    }

    @Test("ACP EOF fails an in-flight RPC immediately and the next session can bind")
    func wireEOF() async throws {
        let agent = WorkAgent()
        agent.acceptSessionForTesting("session-a", folder: URL(fileURLWithPath: "/tmp/a"))
        let response = HTTPURLResponse(url: URL(string: "http://127.0.0.1/acp")!, statusCode: 200,
                                       httpVersion: nil, headerFields: nil)!
        agent.transport = ACPTransport(send: { _ in (Data(), response) })
        let pending = Task { () -> Bool in
            do {
                _ = try await agent.call(method: "session/prompt", params: ["sessionId": "session-a"],
                                         session: "session-a", timeout: 900)
                return false
            } catch {
                return true
            }
        }
        try await Task.sleep(for: .milliseconds(50))
        try agent.handleWire(Data(#"{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"session-a","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"partial"}}}}"#.utf8))
        #expect(agent.text == "partial")
        let oldGeneration = agent.lifecycleGeneration
        agent.helperStreamEnded(generation: oldGeneration, session: "session-a")
        #expect(await pending.value)
        #expect(agent.error == "The Chat helper stopped unexpectedly.")
        agent.acceptSessionForTesting("session-b", folder: URL(fileURLWithPath: "/tmp/b"))
        #expect(agent.currentSessionID == "session-b")
        agent.helperStreamEnded(generation: oldGeneration, session: "session-a")
        #expect(agent.currentSessionID == "session-b")
    }

    @Test("Idle helper EOF resets the session without failing the completed turn")
    func idleEOF() {
        let agent = WorkAgent()
        agent.acceptSessionForTesting("session-a", folder: URL(fileURLWithPath: "/tmp/a"))
        var failures = 0
        agent.onChatEvent = { event in
            if case .error = event { failures += 1 }
        }
        agent.helperStreamEnded(generation: agent.lifecycleGeneration, session: "session-a")
        #expect(failures == 0)
        #expect(agent.currentSessionID == nil)
    }

    @Test("Stop invalidates an in-progress preparation generation")
    func preparationCancellation() throws {
        let agent = WorkAgent()
        let generation = agent.lifecycleGeneration
        try agent.checkPreparation(generation)
        agent.stop()
        #expect(throws: CancellationError.self) { try agent.checkPreparation(generation) }
    }

    @Test("Stop releases a pending ACP session/new waiter")
    func stopDuringSessionOpen() async throws {
        let agent = WorkAgent()
        let response = HTTPURLResponse(url: URL(string: "http://127.0.0.1/acp")!, statusCode: 200,
                                       httpVersion: nil, headerFields: nil)!
        agent.transport = ACPTransport(send: { _ in (Data(), response) })
        let pending = Task { () -> Bool in
            do {
                _ = try await agent.call(method: "session/new", params: ["cwd": "/tmp/a"], timeout: 90)
                return false
            } catch { return true }
        }
        try await Task.sleep(for: .milliseconds(50))
        agent.stop()
        #expect(await pending.value)
        try agent.handleWire(Data(#"{"jsonrpc":"2.0","id":1,"result":{"sessionId":"old-session"}}"#.utf8))
        #expect(agent.currentSessionID == nil)
    }

    @Test("Folder selection is frozen while a turn is starting")
    func folderDuringTurn() {
        let model = ChatViewModel(persistsConversations: false)
        let original = model.gooseFolderPath
        model.send("Build it", alias: "test-model")
        model.setGooseFolder(URL(fileURLWithPath: "/tmp/other-goose-folder"))
        #expect(model.gooseFolderPath == original)
        model.stop()
    }

    @Test("Duplicate ACP permission RPC is queued once and Stop invalidates it")
    func wirePermission() throws {
        let agent = WorkAgent()
        agent.acceptSessionForTesting("session-a", folder: URL(fileURLWithPath: "/tmp/a"))
        let packet = #"{"jsonrpc":"2.0","id":42,"method":"session/request_permission","params":{"sessionId":"session-a","toolCall":{"toolCallId":"t1","title":"Run command","rawInput":{"command":"echo hi"}},"options":[{"kind":"allow_once","optionId":"yes"},{"kind":"reject_once","optionId":"no"}]}}"#
        try agent.handleWire(Data(packet.utf8))
        try agent.handleWire(Data(packet.utf8))
        #expect(agent.permission?.id == "42")
        agent.stop()
        #expect(agent.permission == nil)
        try agent.handleWire(Data(packet.utf8))
        #expect(agent.permission == nil)
    }

    @Test("Switching models closes a pending approval card")
    func modelSwitchWithApproval() {
        let model = ChatViewModel(persistsConversations: false)
        model.devSeedMessages([ChatMessage(role: .user, content: "Run it", status: .complete),
                               ChatMessage(role: .assistant, status: .streaming)])
        let request = WorkPermission(id: "42", sessionID: "old-session", toolCallID: "t1",
                                     command: "echo hi", options: [])
        model.applyGooseEvent(.tool(id: "t1", name: "developer__shell", input: "echo hi",
                                    status: "in_progress", output: ""), initialPlaceholder: 1)
        model.applyGooseEvent(.permission(request, "developer__shell", "permission needed"), initialPlaceholder: 1)
        #expect(model.goosePermission(for: "t1") != nil)
        model.selectedGooseModelChanged()
        #expect(model.goosePermission(for: "t1") == nil)
        #expect(model.messages[1].gooseParts?.first?.tool?.permissionState == "cancelled")
        #expect(model.messages[1].gooseParts?.first?.tool?.status == "cancelled")
    }

    @Test("Goose Chat isolates global skills and keeps a compact tool set")
    func isolatedHelperContext() {
        #expect(WorkAgent.helperHome(gooseChat: true, fullAccess: true,
            userHome: "/Users/test", support: "/tmp/chat-support") == "/tmp/chat-support")
        #expect(WorkAgent.builtinExtensions(gooseChat: true, fullAccess: true) == "developer,computercontroller,memory,autovisualiser,tutorial")
        #expect(WorkAgent.helperHome(gooseChat: false, fullAccess: true,
            userHome: "/Users/test", support: "/tmp/chat-support") == "/Users/test")
    }

    @Test("Tool titles use verbs and paths relative to the chat folder")
    func toolTitles() {
        let folder = "/tmp/chat"
        #expect(GooseToolPresentation.title(name: "developer__write_file", input: #"{"path":"/tmp/chat/snake.html"}"#, folder: folder) == "Wrote snake.html")
        #expect(GooseToolPresentation.title(name: "developer__write_file", input: #"{"path":"/tmp/chat/games/snake.html"}"#, folder: folder) == "Wrote games/snake.html")
        #expect(GooseToolPresentation.title(name: "write · /tmp/chat/games/snake.html", input: #"{"content":"hi"}"#, folder: folder) == "Wrote games/snake.html")
        #expect(GooseToolPresentation.title(name: "developer__edit_file", input: #"{"path":"/tmp/chat/snake.html"}"#, folder: folder) == "Edited snake.html")
        #expect(GooseToolPresentation.title(name: "developer__shell", input: #"{"command":"python3 hello.py"}"#, folder: folder) == "Ran `python3 hello.py`")
        #expect(GooseToolPresentation.title(name: "developer__list", input: "{}", folder: folder) == "Listed files")
        #expect(GooseToolPresentation.title(name: "computercontroller__open", input: #"{"path":"/tmp/chat/snake.html"}"#, folder: folder) == "Opened snake.html")
        #expect(GooseToolPresentation.title(name: "developer__read", input: #"{"path":"/tmp/chat/snake.html"}"#, folder: folder) == "Read snake.html")
        #expect(GooseToolPresentation.title(name: "extension__custom", input: #"{}"#, folder: folder) == "custom")
        #expect(GooseToolPresentation.title(name: "custom", input: #"{"path":"/tmp/chat/snake.html"}"#, folder: folder) == "custom · snake.html")
        #expect(GooseToolPresentation.outputPreview((1...45).map(String.init).joined(separator: "\n"), expanded: false).components(separatedBy: .newlines).count == 40)
    }

    @Test("Progress names the current step and advances through tool updates")
    func progressMapping() {
        #expect(GooseToolPresentation.progress(card: nil, elapsed: 45, folder: "/tmp/chat") == "Preparing response · 45 s")
        let card = GooseToolCard(id: "t", name: "developer__write_file",
            input: #"{"path":"/tmp/chat/snake.html","content":"hello"}"#, status: "running")
        #expect(GooseToolPresentation.progress(card: card, elapsed: 12, folder: "/tmp/chat").contains("Writing snake.html"))
        #expect(GooseToolPresentation.isRunning("in_progress"))
        #expect(GooseToolPresentation.isFailure("failed"))
    }

    @Test("Completed edits expose only files inside the conversation folder")
    func producedFile() {
        let card = GooseToolCard(id: "t", name: "developer__write_file",
            input: #"{"path":"/tmp/chat/snake.html"}"#, status: "completed")
        #expect(GooseToolPresentation.producedFile(card, folder: "/tmp/chat")?.path == "/tmp/chat/snake.html")
        var titled = card
        titled.name = "write · /tmp/chat/games/snake.html"
        titled.input = #"{"content":"hi"}"#
        #expect(GooseToolPresentation.producedFile(titled, folder: "/tmp/chat")?.path == "/tmp/chat/games/snake.html")
        titled.folder = "/tmp/chat"
        let restored = try? JSONDecoder().decode(GooseToolCard.self, from: JSONEncoder().encode(titled))
        #expect(GooseToolPresentation.producedFile(restored!, folder: restored!.folder!)?.path == "/tmp/chat/games/snake.html")
        var running = card
        running.status = "running"
        #expect(GooseToolPresentation.producedFile(running, folder: "/tmp/chat") == nil)
        var outside = card
        outside.input = #"{"path":"/tmp/other/private.html"}"#
        #expect(GooseToolPresentation.producedFile(outside, folder: "/tmp/chat") == nil)
    }

    @Test("Goose only continues an output-limited turn")
    func continuationPolicy() {
        let failed = [WorkStep(id: "tool", command: "write", status: "failed")]
        #expect(!WorkAgent.shouldContinue(gooseChat: true,
            result: ["stopReason": "end_turn"], newText: "Done", newSteps: failed))
        #expect(WorkAgent.shouldContinue(gooseChat: true,
            result: ["stopReason": "max_tokens"], newText: "Partial", newSteps: []))
        #expect(WorkAgent.shouldContinue(gooseChat: false,
            result: ["stopReason": "end_turn"], newText: "Done", newSteps: failed))
    }

    @Test("Permission policy keeps decisions in the app")
    func permissionPolicy() {
        let defaults = UserDefaults(suiteName: "GooseChatTests.\(UUID().uuidString)")!
        defaults.set(true, forKey: GoosePermissionPolicy.yoloKey)
        GoosePermissionPolicy.resetYOLOForLaunch(defaults: defaults)
        #expect(!defaults.bool(forKey: GoosePermissionPolicy.yoloKey))
        #expect(GoosePermissionPolicy.decide(yolo: false, alwaysAllowed: [], tool: "developer").answer == .ask)
        #expect(GoosePermissionPolicy.decide(yolo: true, alwaysAllowed: [], tool: "developer").cardState == "auto-allowed")
        #expect(GoosePermissionPolicy.decide(yolo: false, alwaysAllowed: ["developer__shell"], tool: "developer__shell").answer == .allow)
        #expect(GoosePermissionPolicy.decide(yolo: false, alwaysAllowed: [], tool: "developer", userDecision: .allow).cardState == "allowed")
        #expect(GoosePermissionPolicy.decide(yolo: false, alwaysAllowed: [], tool: "developer", userDecision: .decline).answer == .decline)
        let request = WorkPermission(id: "42", sessionID: "s", toolCallID: "t", command: "run", options: [], rpcID: 42)
        #expect(request.rpcID as? Int == 42)
        let call: [String: Any] = ["title": "developer: shell · echo secret",
            "_meta": ["goose": ["toolCall": ["toolName": "developer__shell"]]]]
        #expect(GoosePermissionPolicy.toolIdentity(call) == "developer__shell")
        #expect(GoosePermissionPolicy.scope("developer__shell") == "Run shell commands")
        #expect(GoosePermissionPolicy.decide(yolo: false, alwaysAllowed: ["developer__shell"],
            tool: "developer__shell").answer == .allow)
        #expect(GoosePermissionPolicy.decide(yolo: false, alwaysAllowed: ["developer: shell · echo secret"],
            tool: "developer__shell").answer == .ask)
        #expect(WorkAgent.redacted("output: acp-secret and engine-bearer",
            secrets: ["acp-secret", "engine-bearer"]) == "output: [redacted] and [redacted]")
        #expect(WorkAgent.redacted("output: acp-se", secrets: ["acp-secret"]) == "output: [redacted]")
    }

    @Test("Native grants are replaced at launch even after a YOLO tool writes them")
    func nativePermissionReset() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("goose-policy-\(UUID())")
        let config = root.appendingPathComponent("config")
        try FileManager.default.createDirectory(at: config, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let policy = config.appendingPathComponent("permission.yaml")
        let defaults = UserDefaults(suiteName: "GooseGrantRestart.\(UUID())")!
        defaults.set(true, forKey: GoosePermissionPolicy.yoloKey)
        #expect(GoosePermissionPolicy.decide(tool: "developer__shell", defaults: defaults).answer == .allow)
        try "user:\n  always_allow: [developer__shell]\n".write(to: policy, atomically: true, encoding: .utf8)
        GoosePermissionPolicy.resetYOLOForLaunch(defaults: defaults)
        try GoosePermissionPolicy.resetNativePolicy(in: root)
        #expect(try String(contentsOf: policy, encoding: .utf8) == GoosePermissionPolicy.lockedNativePolicy)
        #expect(GoosePermissionPolicy.decide(tool: "developer__shell", defaults: defaults).answer == .ask)
        let outside = FileManager.default.temporaryDirectory.appendingPathComponent("goose-grant-\(UUID()).yaml")
        defer { try? FileManager.default.removeItem(at: outside) }
        try "outside".write(to: outside, atomically: true, encoding: .utf8)
        try FileManager.default.removeItem(at: policy)
        try FileManager.default.createSymbolicLink(at: policy, withDestinationURL: outside)
        try GoosePermissionPolicy.resetNativePolicy(in: root)
        #expect(try String(contentsOf: policy, encoding: .utf8) == GoosePermissionPolicy.lockedNativePolicy)
        #expect(try String(contentsOf: outside, encoding: .utf8) == "outside")
    }

    @Test("App owned file actions reject symlink escapes")
    func rejectsSymlinkEscape() throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("goose-path-\(UUID())")
        let folder = base.appendingPathComponent("chat")
        let outside = base.appendingPathComponent("outside")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        try FileManager.default.createSymbolicLink(at: folder.appendingPathComponent("Attachments"), withDestinationURL: outside)
        #expect(throws: CocoaError.self) {
            _ = try GooseToolPresentation.containedDestination(folder.appendingPathComponent("Attachments/new.txt"), root: folder)
        }
        let card = GooseToolCard(id: "t", name: "developer__write_file",
            input: "{\"path\":\"\(folder.path)/Attachments/new.txt\"}", status: "completed")
        #expect(GooseToolPresentation.producedFile(card, folder: folder.path) == nil)
    }

    @Test("ACP text, thinking, tool lifecycle and permission enter the transcript")
    func eventMapping() {
        let model = ChatViewModel(persistsConversations: false)
        model.devSeedMessages([ChatMessage(role: .user, content: "Build it", status: .complete),
                               ChatMessage(role: .assistant, status: .streaming)])
        model.applyGooseEvent(.text("Hello"), initialPlaceholder: 1)
        model.applyGooseEvent(.thinking("plan"), initialPlaceholder: 1)
        #expect(model.messages[1].content == "Hello")
        #expect(model.messages[1].reasoning == "plan")
        model.applyGooseEvent(.tool(id: "t1", name: "developer", input: "write", status: "running", output: ""), initialPlaceholder: 1)
        model.applyGooseEvent(.permission(WorkPermission(id: "1", sessionID: "s", toolCallID: "t1", command: "write", options: []), "developer", "permission needed"), initialPlaceholder: 1)
        model.applyGooseEvent(.tool(id: "t1", name: "developer", input: "", status: "completed", output: "done"), initialPlaceholder: 1)
        model.applyGooseEvent(.tool(id: "t1", name: "write · /tmp/chat/report.txt", input: "", status: "completed", output: ""), initialPlaceholder: 1)
        model.applyGooseEvent(.tool(id: "t1", name: "", input: "", status: "completed", output: ""), initialPlaceholder: 1)
        #expect(model.messages.count == 2)
        #expect(model.messages[1].gooseParts?.count == 1)
        #expect(model.messages[1].gooseParts?[0].tool?.output == "done")
        #expect(model.messages[1].gooseParts?[0].tool?.name == "write · /tmp/chat/report.txt")
        #expect(model.messages[1].gooseParts?[0].tool?.permissionState == "permission needed")
        model.applyGooseEvent(.text("Done"), initialPlaceholder: 1)
        model.applyGooseEvent(.throughput(42.5), initialPlaceholder: 1)
        #expect(model.messages[1].stats?.reportedTokensPerSecond == 42.5)
        #expect(model.messages.count == 2)
        #expect(model.messages[1].content == "HelloDone")
        #expect(model.messages[1].gooseParts?.count == 1)
        #expect(model.messages[1].gooseParts?.allSatisfy {
            $0.tool != nil || $0.text?.isEmpty == false
        } == true)
        model.applyGooseEvent(.error("Server error (502 Bad Gateway) at http://127.0.0.1:1234"), initialPlaceholder: 1)
        #expect(model.lastError == ChatViewModel.gooseFailureCopy)
        #expect(model.messages[1].errorMessage == ChatViewModel.gooseFailureCopy)
        #expect(model.messages[1].gooseErrorDetails?.contains("502 Bad Gateway") == true)
        #expect(model.messages.last?.status == .failed)
        model.applyGooseEvent(.cancelled, initialPlaceholder: 1)
        let textError = ChatViewModel(persistsConversations: false)
        textError.devSeedMessages([ChatMessage(role: .user, content: "hi"),
                                   ChatMessage(role: .assistant, status: .streaming)])
        textError.applyGooseEvent(.text("Ran into this error: Server error"), initialPlaceholder: 1)
        textError.applyGooseEvent(.text(" (502 Bad Gateway) at http://127.0.0.1:57718"), initialPlaceholder: 1)
        #expect(textError.messages[1].content.isEmpty)
        #expect(textError.messages[1].status == .failed)
        #expect(textError.messages[1].gooseErrorDetails?.contains("57718") == true)
    }

    @Test("Stored conversation retains session and folder; replay is bounded")
    func reopen() throws {
        let conversation = ChatConversation(id: UUID(), title: "Task", messages: [ChatMessage(role: .user, content: "first", status: .complete)], createdAt: .now, updatedAt: .now, gooseSessionID: "session-1", gooseFolder: "/tmp/scratch")
        let decoded = try JSONDecoder().decode(ChatConversation.self, from: JSONEncoder().encode(conversation))
        #expect(decoded.gooseSessionID == "session-1")
        #expect(decoded.gooseFolder == "/tmp/scratch")
        #expect(ChatViewModel.repairGooseTitles([decoded]).first?.title == "first")
        var renamed = decoded
        renamed.hasCustomTitle = true
        #expect(ChatViewModel.repairGooseTitles([renamed]).first?.title == "Task")
        var legacy = decoded
        legacy.messages.append(ChatMessage(role: .assistant,
            content: "Ran into this error: Server error (502 Bad Gateway) at http://127.0.0.1:57718"))
        let repaired = ChatViewModel.repairGooseTitles([legacy])[0].messages.last!
        #expect(repaired.content.isEmpty)
        #expect(repaired.errorMessage == ChatViewModel.gooseFailureCopy)
        #expect(repaired.gooseErrorDetails?.contains("57718") == true)
        legacy.branches = [legacy.messages.last!]
        #expect(ChatViewModel.repairGooseTitles([legacy])[0].branches.first?.content.isEmpty == true)
        var quoted = decoded
        quoted.messages.append(ChatMessage(role: .assistant,
            content: "Ran into this error: please check your input"))
        #expect(ChatViewModel.repairGooseTitles([quoted])[0].messages.last?.content
            == "Ran into this error: please check your input")
        #expect(ChatViewModel.gooseReplay(decoded.messages).contains("User [completed]: first"))
        let toolOnly = ChatMessage(role: .assistant, gooseParts: [
            GooseTurnPart(tool: GooseToolCard(id: "t", name: "write · snake.html",
                input: "{}", output: "created", status: "completed"))])
        #expect(ChatViewModel.gooseReplay(decoded.messages + [toolOnly]).contains("Tools: write · snake.html: completed"))
        let longAnswer = ChatMessage(role: .assistant, content: String(repeating: "preamble ", count: 180) + "Final answer: done")
        #expect(ChatViewModel.gooseReplay(decoded.messages + [longAnswer]).contains("Final answer: done"))
        var cancelled = ChatMessage(role: .assistant, content: "partial")
        cancelled.errorMessage = "Stopped."
        var failed = ChatMessage(role: .assistant, content: "broken", status: .failed)
        failed.errorMessage = "Failed"
        let legacyTool = ChatMessage(role: .tool, content: "Older tool output", toolCallID: "old")
        let replay = ChatViewModel.gooseReplay(decoded.messages + [cancelled, failed, legacyTool])
        #expect(replay.contains("[cancelled]"))
        #expect(replay.contains("[failed]"))
        #expect(replay.contains("Tool result [completed]: Older tool output"))
        #expect(replay.contains("Bounded reconstruction"))
        let visible = decoded.messages.map(\.id)
        #expect(!ChatViewModel.canReuseGooseSession(activeConversationID: decoded.id,
            sessionConversationID: nil, currentContextIDs: visible, visibleContextIDs: visible, sessionID: nil))
        #expect(ChatViewModel.canReuseGooseSession(activeConversationID: decoded.id,
            sessionConversationID: decoded.id, currentContextIDs: visible, visibleContextIDs: visible, sessionID: "s"))
        #expect(!ChatViewModel.canReuseGooseSession(activeConversationID: decoded.id,
            sessionConversationID: decoded.id, currentContextIDs: visible, visibleContextIDs: [UUID()], sessionID: "s"))
    }

    @Test("Deletion removes only owned copies and all recorded helper sessions")
    func retention() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("goose-retain-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let folder = root.appendingPathComponent("workspace")
        let user = ChatMessage(role: .user, content: "", fileAttachments: [
            try ChatFileAttachment(filename: "source.csv", kind: .csv, extractedText: "preview",
                                   sourceByteCount: 7)])
        let copies = folder.appendingPathComponent("Attachments/\(user.id.uuidString)")
        try FileManager.default.createDirectory(at: copies, withIntermediateDirectories: true)
        let owned = copies.appendingPathComponent(user.fileAttachments[0].id.uuidString + ".txt")
        let userOutput = copies.appendingPathComponent("my-result.txt")
        try "copy".write(to: owned, atomically: true, encoding: .utf8)
        try "keep".write(to: userOutput, atomically: true, encoding: .utf8)
        let copiedHash = SHA256.hash(data: Data("copy".utf8))
            .map { String(format: "%02x", $0) }.joined()
        try GooseConversationRetention.deleteAttachmentDirectories([copies.path],
            folders: [folder.path], messages: [user], hashes: [owned.path: copiedHash])
        #expect(!FileManager.default.fileExists(atPath: owned.path))
        #expect(FileManager.default.fileExists(atPath: userOutput.path))
        try "changed by user".write(to: owned, atomically: true, encoding: .utf8)
        try GooseConversationRetention.deleteAttachmentDirectories([copies.path],
            folders: [folder.path], messages: [user], hashes: [owned.path: copiedHash])
        #expect(FileManager.default.fileExists(atPath: owned.path))
        let legacyCopy = copies.appendingPathComponent("source.txt")
        try "preview".write(to: legacyCopy, atomically: true, encoding: .utf8)
        try GooseConversationRetention.deleteAttachmentDirectories([copies.path],
            folders: [folder.path], messages: [user])
        #expect(!FileManager.default.fileExists(atPath: legacyCopy.path))
        #expect(FileManager.default.fileExists(atPath: owned.path))

        let hash = SHA256.hash(data: Data(folder.path.utf8))
            .map { String(format: "%02x", $0) }.joined()
        let database = root.appendingPathComponent("databases/\(hash)/data/sessions/sessions.db")
        try FileManager.default.createDirectory(at: database.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        var handle: OpaquePointer?
        #expect(sqlite3_open(database.path, &handle) == SQLITE_OK)
        defer { sqlite3_close(handle) }
        #expect(sqlite3_exec(handle, "CREATE TABLE sessions(id TEXT PRIMARY KEY); "
            + "CREATE TABLE messages(session_id TEXT); CREATE TABLE usage_ledger(session_id TEXT); "
            + "INSERT INTO sessions VALUES('first'),('superseded'),('other'); "
            + "INSERT INTO messages VALUES('first'),('superseded'),('other'); "
            + "INSERT INTO usage_ledger VALUES('first'),('superseded'),('other');",
            nil, nil, nil) == SQLITE_OK)
        try GooseConversationRetention.deleteSessions([
            GooseOwnedSession(id: "first", folder: folder.path),
            GooseOwnedSession(id: "superseded", folder: folder.path),
        ], root: root.appendingPathComponent("databases"))
        var statement: OpaquePointer?
        #expect(sqlite3_prepare_v2(handle, "SELECT count(*) FROM sessions", -1, &statement, nil) == SQLITE_OK)
        #expect(sqlite3_step(statement) == SQLITE_ROW)
        #expect(sqlite3_column_int(statement, 0) == 1)
        sqlite3_finalize(statement)
        let contract = root.appendingPathComponent("databases/\(hash)/chat-contract.json")
        let conversationID = UUID()
        try JSONSerialization.data(withJSONObject: ["conversation_id": conversationID.uuidString,
            "system_prompt": "private memory"]).write(to: contract)
        try GooseConversationRetention.deleteContractsOwned(by: UUID(),
            folders: [folder.path], root: root.appendingPathComponent("databases"))
        #expect(FileManager.default.fileExists(atPath: contract.path))
        try GooseConversationRetention.deleteContractsOwned(by: conversationID,
            folders: [folder.path], root: root.appendingPathComponent("databases"))
        #expect(!FileManager.default.fileExists(atPath: contract.path))
    }

    @Test("Goose exports the retained extract and replays attachment-only rows")
    func retainedAttachment() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("goose-extract-\(UUID())")
        defer { try? FileManager.default.removeItem(at: folder) }
        let cache = DocumentContentCache(diskDirectory: nil)
        let file = try ChatFileAttachment(filename: "large.csv", kind: .csv,
            extractedText: "opening preview", sourceByteCount: 100, totalCharacterCount: 28)
        cache.put(file.id, entry: DocumentContentCache.Entry(
            filename: file.filename, text: "opening preview\nfinal row,42", isComplete: true))
        let model = ChatViewModel(persistsConversations: false, documentCache: cache)
        model.setGooseFolder(folder)
        let user = ChatMessage(role: .user, fileAttachments: [file])
        let note = try await model.gooseAttachmentNote(for: user)
        let exported = folder.appendingPathComponent("Attachments/\(user.id.uuidString)/\(file.id.uuidString).txt")
        #expect(note.contains(exported.path))
        #expect(try String(contentsOf: exported, encoding: .utf8).contains("Extraction status: complete"))
        #expect(try String(contentsOf: exported, encoding: .utf8).contains("final row,42"))
        let replay = ChatViewModel.gooseReplay([user])
        #expect(replay.contains("Attachments/\(user.id.uuidString)/\(file.id.uuidString).txt"))
        cache.remove(file.id)
        _ = try await model.gooseAttachmentNote(for: user)
        #expect(try String(contentsOf: exported, encoding: .utf8).contains("final row,42"))
        try "user changed this copy".write(to: exported, atomically: true, encoding: .utf8)
        _ = try await model.gooseAttachmentNote(for: user)
        #expect(try String(contentsOf: exported, encoding: .utf8) == "user changed this copy")
    }

    @Test("A conversation switch during extraction cannot export into the next chat")
    func extractionSwitch() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("goose-switch-\(UUID())")
        defer { try? FileManager.default.removeItem(at: folder) }
        let cache = DocumentContentCache(diskDirectory: nil)
        let file = try ChatFileAttachment(filename: "pending.txt", kind: .txt,
            extractedText: "preview", sourceByteCount: 7)
        cache.put(file.id, entry: DocumentContentCache.Entry(
            filename: file.filename, text: "full text", isComplete: true))
        cache.beginPending(file.id)
        let model = ChatViewModel(persistsConversations: false, documentCache: cache)
        model.setGooseFolder(folder)
        let user = ChatMessage(role: .user, fileAttachments: [file])
        let task = Task { try await model.gooseAttachmentNote(for: user) }
        let directory = folder.appendingPathComponent("Attachments/\(user.id.uuidString)")
        for _ in 0..<100 where !FileManager.default.fileExists(atPath: directory.path) {
            try await Task.sleep(for: .milliseconds(2))
        }
        model.newConversation()
        cache.finishPending(file.id)
        _ = try await task.value
        #expect(!FileManager.default.fileExists(atPath:
            directory.appendingPathComponent(file.id.uuidString + ".txt").path))
    }

    @Test("Replay stops before a second attachment after a conversation switch")
    func replayStopsAfterSwitch() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("goose-replay-switch-\(UUID())")
        defer { try? FileManager.default.removeItem(at: folder) }
        let cache = DocumentContentCache(diskDirectory: nil)
        let firstFile = try ChatFileAttachment(filename: "pending.txt", kind: .txt,
            extractedText: "preview", sourceByteCount: 7)
        let secondFile = try ChatFileAttachment(filename: "next.txt", kind: .txt,
            extractedText: "preview", sourceByteCount: 7)
        cache.put(firstFile.id, entry: DocumentContentCache.Entry(
            filename: firstFile.filename, text: "full text", isComplete: true))
        cache.beginPending(firstFile.id)
        let model = ChatViewModel(persistsConversations: false, documentCache: cache)
        model.setGooseFolder(folder)
        let first = ChatMessage(role: .user, fileAttachments: [firstFile])
        let second = ChatMessage(role: .user, fileAttachments: [secondFile])
        let ownerID = model.activeConversationID
        let task = Task { try await model.replayGooseAttachments([first, second], ownerID: ownerID) }
        let firstDirectory = folder.appendingPathComponent("Attachments/\(first.id.uuidString)")
        for _ in 0..<100 where !FileManager.default.fileExists(atPath: firstDirectory.path) {
            try await Task.sleep(for: .milliseconds(2))
        }
        model.newConversation()
        let nextOwnerID = model.activeConversationID
        cache.finishPending(firstFile.id)
        try await task.value
        #expect(model.activeConversationID == nextOwnerID)
        #expect(!FileManager.default.fileExists(atPath:
            folder.appendingPathComponent("Attachments/\(second.id.uuidString)").path))
    }

    @Test("An edit made during extraction is not overwritten")
    func editDuringExtraction() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("goose-edit-\(UUID())")
        defer { try? FileManager.default.removeItem(at: folder) }
        let cache = DocumentContentCache(diskDirectory: nil)
        let file = try ChatFileAttachment(filename: "pending.txt", kind: .txt,
            extractedText: "preview", sourceByteCount: 7)
        cache.put(file.id, entry: DocumentContentCache.Entry(filename: file.filename,
            text: "retained text", isComplete: true))
        let model = ChatViewModel(persistsConversations: false, documentCache: cache)
        model.setGooseFolder(folder)
        let user = ChatMessage(role: .user, fileAttachments: [file])
        _ = try await model.gooseAttachmentNote(for: user)
        let exported = folder.appendingPathComponent("Attachments/\(user.id.uuidString)/\(file.id.uuidString).txt")
        cache.beginPending(file.id)
        let task = Task { try await model.gooseAttachmentNote(for: user) }
        try await Task.sleep(for: .milliseconds(10))
        try "user output".write(to: exported, atomically: true, encoding: .utf8)
        cache.finishPending(file.id)
        _ = try await task.value
        #expect(try String(contentsOf: exported, encoding: .utf8) == "user output")
    }

    @Test("Goose contract carries explicit instructions and sampling, without Desktop memory")
    func requestContract() {
        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let sampling = ResolvedSampling(temperature: 0.23, topP: 0.95,
            maxTokens: 1536, repetitionPenalty: 1.1,
            repetitionPenaltyIsImplicitDefault: true, enableThinking: false)
        let expected = ChatViewModel.addingInstructionLayers(to: [],
            global: "Be concise", conversation: "Use Swift").first?.content
        let contract = ChatViewModel.gooseRequestContract(
            globalInstruction: "Be concise", conversationInstruction: "Use Swift",
            sampling: sampling, alias: "test-model", now: date)
        #expect(contract["system_prompt"] as? String == expected)
        #expect(contract["temperature"] as? Double == 0.23)
        #expect(contract["top_p"] as? Double == 0.95)
        #expect(contract["max_tokens"] as? Int == 1536)
        #expect(contract["repetition_penalty"] as? Double == 1.1)
        #expect(contract["enable_thinking"] as? Bool == false)
    }

    @Test("A successful Goose reply delivers product value exactly once")
    func productValue() {
        var delivered = 0
        let model = ChatViewModel(persistsConversations: false,
            onProductValueDelivered: { kind in
                if kind == .chatReply { delivered += 1 }
            })
        let reply = ChatMessage(role: .assistant, content: "Done", status: .complete)
        model.reportGooseProductValueIfDelivered(reply)
        model.reportGooseProductValueIfDelivered(reply)
        model.reportGooseProductValueIfDelivered(ChatMessage(role: .assistant,
            content: "partial", status: .failed))
        #expect(delivered == 1)
    }

    @Test("Goose chats appear newest first and reopen their saved transcript")
    func historyList() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("goose-history-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let store = root.appendingPathComponent("conversations.json")
        let model = ChatViewModel(conversationStoreURL: store)
        model.send("First task", alias: "test-model")
        model.stopAndPersist()
        let firstID = model.activeConversationID
        model.newConversation()
        model.send("Second task", alias: "test-model")
        model.stopAndPersist()
        let secondID = model.activeConversationID
        #expect(model.conversations.map(\.id) == [secondID, firstID])
        #expect(model.conversations.map(\.title) == ["Second task", "First task"])
        model.selectConversation(firstID)
        #expect(model.messages.first?.content == "First task")
        #expect(model.activeConversationID == firstID)
        model.send("Continue", alias: "test-model")
        model.stopAndPersist()
        #expect(model.conversations.first?.id == firstID)
        #expect(model.messages.contains(where: { $0.content == "Continue" }))
    }

    @Test("Helper failure activates direct chat fallback and Retry clears it")
    func fallback() {
        let model = ChatViewModel(persistsConversations: false)
        model.activateGooseFallback(WorkError.message("missing helper"))
        #expect(model.gooseFallback != nil)
        model.retryGoose()
        #expect(model.gooseFallback == nil)
    }
}
