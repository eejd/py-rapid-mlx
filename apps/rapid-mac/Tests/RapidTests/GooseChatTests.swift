import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Goose Chat")
struct GooseChatTests {
    @Test("Goose Chat isolates global skills and keeps a compact tool set")
    func isolatedHelperContext() {
        #expect(WorkAgent.helperHome(gooseChat: true, fullAccess: true,
            userHome: "/Users/test", support: "/tmp/chat-support") == "/tmp/chat-support")
        #expect(WorkAgent.builtinExtensions(gooseChat: true, fullAccess: true) == "developer")
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

    @Test("Goose behavior stays within two lines and ends with the user request")
    func behaviorPrompt() {
        let behavior = WorkAgent.gooseChatBehavior
        #expect(behavior.split(separator: "\n").count == 2)
        #expect(behavior.contains("Reply in the user's language"))
        #expect(behavior.contains("do not re-run or re-verify"))
        #expect(behavior.contains("paths relative to the selected chat folder"))
        #expect((behavior + "請寫程式").hasSuffix("請寫程式"))
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
        #expect(GoosePermissionPolicy.decide(yolo: false, alwaysAllowed: ["developer"], tool: "developer").answer == .allow)
        #expect(GoosePermissionPolicy.decide(yolo: false, alwaysAllowed: [], tool: "developer", userDecision: .allow).cardState == "allowed")
        #expect(GoosePermissionPolicy.decide(yolo: false, alwaysAllowed: [], tool: "developer", userDecision: .decline).answer == .decline)
        let request = WorkPermission(id: "42", sessionID: "s", toolCallID: "t", command: "run", options: [], rpcID: 42)
        #expect(request.rpcID as? Int == 42)
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
        #expect(model.messages[1].gooseParts?.count == 2)
        #expect(model.messages[1].gooseParts?[1].tool?.output == "done")
        #expect(model.messages[1].gooseParts?[1].tool?.name == "write · /tmp/chat/report.txt")
        #expect(model.messages[1].gooseParts?[1].tool?.permissionState == "permission needed")
        model.applyGooseEvent(.text("Done"), initialPlaceholder: 1)
        model.applyGooseEvent(.throughput(42.5), initialPlaceholder: 1)
        #expect(model.messages[1].stats?.reportedTokensPerSecond == 42.5)
        #expect(model.messages.count == 2)
        #expect(model.messages[1].content == "HelloDone")
        #expect(model.messages[1].gooseParts?.count == 3)
        #expect(model.messages[1].gooseParts?[2].text == "Done")
        #expect(model.messages[1].gooseParts?.allSatisfy {
            $0.tool != nil || $0.text?.isEmpty == false
        } == true)
        model.applyGooseEvent(.error("Server error (502 Bad Gateway) at http://127.0.0.1:1234"), initialPlaceholder: 1)
        #expect(model.lastError == ChatViewModel.gooseFailureCopy)
        #expect(model.messages[1].errorMessage == ChatViewModel.gooseFailureCopy)
        #expect(model.messages[1].gooseErrorDetails?.contains("502 Bad Gateway") == true)
        #expect(model.messages.last?.status == .failed)
        model.applyGooseEvent(.cancelled, initialPlaceholder: 1)
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
        #expect(ChatViewModel.gooseReplay(decoded.messages).contains("User: first"))
        let toolOnly = ChatMessage(role: .assistant, gooseParts: [
            GooseTurnPart(tool: GooseToolCard(id: "t", name: "write · snake.html",
                input: "{}", output: "created", status: "completed"))])
        #expect(ChatViewModel.gooseReplay(decoded.messages + [toolOnly]).contains("Tools: write · snake.html: completed"))
        let longAnswer = ChatMessage(role: .assistant, content: String(repeating: "preamble ", count: 180) + "Final answer: done")
        #expect(ChatViewModel.gooseReplay(decoded.messages + [longAnswer]).contains("Final answer: done"))
        let visible = decoded.messages.map(\.id)
        #expect(!ChatViewModel.canReuseGooseSession(activeConversationID: decoded.id,
            sessionConversationID: nil, currentContextIDs: visible, visibleContextIDs: visible, sessionID: nil))
        #expect(ChatViewModel.canReuseGooseSession(activeConversationID: decoded.id,
            sessionConversationID: decoded.id, currentContextIDs: visible, visibleContextIDs: visible, sessionID: "s"))
        #expect(!ChatViewModel.canReuseGooseSession(activeConversationID: decoded.id,
            sessionConversationID: decoded.id, currentContextIDs: visible, visibleContextIDs: [UUID()], sessionID: "s"))
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
