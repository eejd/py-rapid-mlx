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
}
