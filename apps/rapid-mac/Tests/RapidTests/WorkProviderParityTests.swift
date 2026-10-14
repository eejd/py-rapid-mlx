import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Work Full provider parity")
struct WorkProviderParityTests {
    @Test("uses the selected model's advertised context rather than a fixed 16K limit")
    func selectedContextWindow() throws {
        let response = #"{"data":[{"id":"other","context_window":8192},{"id":"qwen","context_window":262144}]}"#
        let data = try #require(response.data(using: .utf8))
        #expect(WorkAgent.contextLimit(from: data, model: "qwen") == 262144)
        #expect(WorkAgent.contextLimit(from: data, model: "other") == 8192)
        #expect(WorkAgent.contextLimit(from: data, model: "missing") == nil)
    }

    @Test("retries only a truncated turn")
    func truncatedTurnContinuation() {
        #expect(WorkAgent.needsContinuation(result: ["stopReason": "max_tokens"], newText: ""))
        #expect(WorkAgent.needsContinuation(result: ["stopReason": "end_turn"],
                                            newText: "Response stopped because the model reached its output-token limit."))
        #expect(!WorkAgent.needsContinuation(result: ["stopReason": "end_turn"], newText: "Done."))
    }

    @Test("checks unfinished work after a failed tool call")
    func failedToolFollowUp() {
        let missingSource = WorkStep(id: "1", command: "cat expenses.csv", status: "failed",
                                     output: "No such file or directory")
        let finished = WorkStep(id: "2", command: "write summary.md", status: "completed")
        #expect(WorkAgent.needsFollowUp(result: ["stopReason": "end_turn"], newText: "Done.",
                                        newSteps: [missingSource, finished]))
        #expect(!WorkAgent.needsFollowUp(result: ["stopReason": "end_turn"], newText: "Done.",
                                         newSteps: [finished]))
    }
}
