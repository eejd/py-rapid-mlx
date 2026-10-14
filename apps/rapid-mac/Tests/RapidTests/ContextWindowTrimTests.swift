import Foundation
import Testing
@testable import Rapid

/// v0.5.11 regression pin for ``ChatViewModel.trimMessagesForContextWindow``.
///
/// Replaces the v0.4.19 cumulative-token meter chip. ChatGPT / Claude
/// desktops never show users a token meter — they drop oldest turns
/// behind the scenes when the conversation would exceed the model's
/// context window. rapid-mlx's server doesn't enforce a window either
/// (mlx-lm RoPE-extrapolates past training context and degrades
/// quality silently), so the client takes responsibility.
///
/// Contract pinned by this suite:
///   * A missing context window or a body under budget is unchanged.
///   * When over budget, oldest turns are dropped while the current user turn
///     and a leading system row remain available.
@MainActor
@Suite("v0.5.11 silent context-window trim")
struct ContextWindowTrimTests {

    // MARK: - No-op cases

    @Test("Nil context window returns the input unchanged")
    func nilContextWindowNoOp() {
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: String(repeating: "x", count: 1_000_000)),
            ChatMessage(role: .assistant, content: String(repeating: "y", count: 1_000_000)),
        ]
        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history, contextWindow: nil
        )
        #expect(trimmed.count == history.count)
    }

    @Test("Empty input returns empty")
    func emptyInputNoOp() {
        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            [], contextWindow: 32_768
        )
        #expect(trimmed.isEmpty)
    }

    @Test("Total under budget returns the input unchanged")
    func underBudgetNoOp() {
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "hi"),
            ChatMessage(role: .assistant, content: "hello"),
            ChatMessage(role: .user, content: "how are you?"),
        ]
        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history, contextWindow: 32_768
        )
        #expect(trimmed == history)
    }

    @Test("Locally extracted document text participates in the context budget")
    func documentTextCountsTowardBudget() throws {
        let attachment = try ChatFileAttachment(
            filename: "large.csv",
            kind: .csv,
            extractedText: String(repeating: "x", count: 8_000),
            sourceByteCount: 8_000
        )
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "old question"),
            ChatMessage(role: .assistant, content: "old answer"),
            ChatMessage(role: .user, content: "analyze", fileAttachments: [attachment]),
        ]

        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history,
            contextWindow: 1_024
        )
        #expect(trimmed.count == 1)
        #expect(trimmed.first?.fileAttachments == [attachment])
    }

    // MARK: - Trimming behaviour

    @Test("Oldest turns are dropped when total exceeds the keep fraction")
    func dropsOldestTurnsOverBudget() {
        // Each 4-char ASCII message costs ceil(4 * 0.42) = 2 estimated tokens,
        // so five messages cost 10. A window of 8 gives a budget of
        // 8 * 0.75 = 6 tokens = three messages. The last user turn always
        // survives, so the two oldest drop.
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "AAAA"),       // 2 tok — drops
            ChatMessage(role: .assistant, content: "BBBB"),  // 2 tok — drops
            ChatMessage(role: .user, content: "CCCC"),       // 2 tok — keeps
            ChatMessage(role: .assistant, content: "DDDD"),  // 2 tok — keeps
            ChatMessage(role: .user, content: "EEEE"),       // 2 tok — keeps (always)
        ]
        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history, contextWindow: 8
        )
        #expect(trimmed.count == 3)
        #expect(trimmed.first?.content == "CCCC")
        #expect(trimmed.last?.content == "EEEE")
    }

    @Test("Most recent user turn is always kept even if it alone overshoots the budget")
    func keepsLastUserTurnEvenWhenOversized() {
        // 1-token budget, last user msg is huge — must still be kept.
        let huge = String(repeating: "x", count: 40_000)
        let history: [ChatMessage] = [
            ChatMessage(role: .user, content: "old"),
            ChatMessage(role: .assistant, content: "older"),
            ChatMessage(role: .user, content: huge),
        ]
        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history, contextWindow: 4
        )
        #expect(trimmed.count == 1)
        #expect(trimmed.first?.content == huge)
    }

    @Test("System message is preserved at index 0 across a trim")
    func preservesSystemMessage() {
        let sys = ChatMessage(role: .system, content: "be terse")
        let history: [ChatMessage] = [
            sys,
            ChatMessage(role: .user, content: "AAAA"),
            ChatMessage(role: .assistant, content: "BBBB"),
            ChatMessage(role: .user, content: "CCCC"),
            ChatMessage(role: .assistant, content: "DDDD"),
            ChatMessage(role: .user, content: "EEEE"),
        ]
        let trimmed = ChatViewModel.trimMessagesForContextWindow(
            history, contextWindow: 4
        )
        #expect(trimmed.first?.role == .system)
        #expect(trimmed.first?.content == "be terse")
        #expect(trimmed.last?.content == "EEEE")
    }

}
