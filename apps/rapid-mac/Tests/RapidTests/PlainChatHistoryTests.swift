import Foundation
import Testing
@testable import Rapid

@Suite("Plain chat after agent removal")
struct PlainChatHistoryTests {
    @Test("A restored agent transcript sends only readable chat turns")
    func restoredAgentHistory() {
        let call = ToolCall(id: "old-call", name: "web_search", arguments: "{}")
        let history = [
            ChatMessage(role: .user, content: "What happened?", wireSuffix: "Old tool instructions"),
            ChatMessage(role: .assistant, toolCalls: [call]),
            ChatMessage(role: .tool, content: "Old search result", toolCallID: call.id),
            ChatMessage(role: .assistant, content: "Here is the answer.", toolCalls: [call]),
            ChatMessage(role: .user, content: "Explain more."),
        ]

        let plain = ChatViewModel.plainChatHistory(history)
        #expect(plain.map(\.role) == [.user, .assistant, .user])
        #expect(plain.map(\.content) == ["What happened?", "Here is the answer.", "Explain more."])
        #expect(plain.allSatisfy { $0.toolCalls == nil && $0.toolCallID == nil && $0.wireSuffix == nil })
    }

    @Test("A stopped empty answer does not leave its prompt on the wire")
    func stoppedTurn() {
        let history = [
            ChatMessage(role: .user, content: "Cancelled prompt"),
            ChatMessage(role: .assistant, status: .failed, errorMessage: "Stopped."),
            ChatMessage(role: .user, content: "New prompt"),
        ]
        #expect(ChatViewModel.plainChatHistory(history).map(\.content) == ["New prompt"])
    }
}
