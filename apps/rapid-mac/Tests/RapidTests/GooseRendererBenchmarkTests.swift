import Foundation
import Testing
@testable import Rapid

@Suite("Goose reply renderer benchmark")
@MainActor
struct GooseRendererBenchmarkTests {
    @Test("Long code reply uses bounded main-thread rendering work")
    func longCodeReply() {
        let code = (0..<300).map { "let value\($0) = \($0) // line of generated code\n" }.joined()
        let reply = "```swift\n" + code + "```\n"
        let chunks = stride(from: 256, through: reply.count, by: 256)
            .map { String(reply.prefix($0)) } + [reply]
        let oldStart = ProcessInfo.processInfo.systemUptime
        for chunk in chunks { _ = TextKitMarkdownView.compile(chunk) }
        let oldMS = (ProcessInfo.processInfo.systemUptime - oldStart) * 1_000

        let store = StreamingMarkdownStore()
        let id = UUID()
        let newStart = ProcessInfo.processInfo.systemUptime
        for chunk in chunks {
            store.enqueue(id: id, text: chunk)
            store.advancePresentationFrame(duration: 1.0 / 60.0)
        }
        store.finish(id: id, finalText: reply)
        while store.isPresentationActive {
            store.advancePresentationFrame(duration: 1.0 / 60.0)
        }
        let newMS = (ProcessInfo.processInfo.systemUptime - newStart) * 1_000
        print("GOOSE_RENDER_M3 chars=\(reply.count) updates=\(chunks.count) before_ms=\(oldMS) after_ms=\(newMS)")
        #expect(store.document.receivedSource == reply)
    }
}
