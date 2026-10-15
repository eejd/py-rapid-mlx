import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Goose Chat")
struct GooseChatTests {
    @Test("Permission policy keeps decisions in the app")
    func permissionPolicy() {
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
        #expect(model.messages[2].gooseTool?.output == "done")
        #expect(model.messages[2].gooseTool?.permissionState == "permission needed")
        model.applyGooseEvent(.text("Done"), initialPlaceholder: 1)
        #expect(model.messages.last?.content == "Done")
        model.applyGooseEvent(.error("tool failed"), initialPlaceholder: 1)
        #expect(model.lastError == "tool failed")
        #expect(model.messages.last?.status == .failed)
        model.applyGooseEvent(.cancelled, initialPlaceholder: 1)
    }

    @Test("Stored conversation retains session and folder; replay is bounded")
    func reopen() throws {
        let conversation = ChatConversation(id: UUID(), title: "Task", messages: [ChatMessage(role: .user, content: "first", status: .complete)], createdAt: .now, updatedAt: .now, gooseSessionID: "session-1", gooseFolder: "/tmp/scratch")
        let decoded = try JSONDecoder().decode(ChatConversation.self, from: JSONEncoder().encode(conversation))
        #expect(decoded.gooseSessionID == "session-1")
        #expect(decoded.gooseFolder == "/tmp/scratch")
        #expect(ChatViewModel.gooseReplay(decoded.messages).contains("User: first"))
        let visible = decoded.messages.map(\.id)
        #expect(!ChatViewModel.canReuseGooseSession(activeConversationID: decoded.id,
            sessionConversationID: nil, currentContextIDs: visible, visibleContextIDs: visible, sessionID: nil))
        #expect(ChatViewModel.canReuseGooseSession(activeConversationID: decoded.id,
            sessionConversationID: decoded.id, currentContextIDs: visible, visibleContextIDs: visible, sessionID: "s"))
        #expect(!ChatViewModel.canReuseGooseSession(activeConversationID: decoded.id,
            sessionConversationID: decoded.id, currentContextIDs: visible, visibleContextIDs: [UUID()], sessionID: "s"))
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
