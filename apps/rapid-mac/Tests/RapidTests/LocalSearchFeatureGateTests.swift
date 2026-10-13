import Foundation
import SwiftUI
import Testing
@testable import Rapid

@Suite("Experimental Local Search gate", .serialized)
struct LocalSearchFeatureGateTests {
    @Test("Search is opt-in and persists the choice")
    func defaultsOff() throws {
        let suite = "rapid.local-search-gate-tests.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        #expect(!LocalSearchFeatureConfig.isEnabled(in: defaults))
        defaults.set(true, forKey: LocalSearchFeatureConfig.enabledKey)
        #expect(LocalSearchFeatureConfig.isEnabled(in: defaults))
        defaults.set(false, forKey: LocalSearchFeatureConfig.enabledKey)
        #expect(!LocalSearchFeatureConfig.isEnabled(in: defaults))
    }

    @MainActor
    @Test("Restoring or disabling hidden Search falls back to Chat")
    func restorationFallback() {
        let requested = SidebarSection.harnessRequested(environment: [
            "RAPID_GUI_GOLDEN_MODE": "1", "RAPID_GUI_INITIAL_SECTION": "search"
        ]) ?? .chat
        #expect(ContentView.sectionAfterLocalSearchGateChange(current: requested, enabled: false) == .chat)
        #expect(ContentView.sectionAfterLocalSearchGateChange(current: requested, enabled: true) == .search)
        #expect(ContentView.sectionAfterLocalSearchGateChange(current: .images, enabled: false) == .images)
    }

    @MainActor
    @Test("Full and compact sidebars hide Search by default and expose it only on opt-in")
    func sidebarGate() async throws {
        let chat = ChatViewModel(persistsConversations: false)
        for compact in [false, true] {
            do {
                let stage = GoldenStage(SidebarView(
                    isCompact: compact, selection: .constant(.chat),
                    chat: chat, onNewChat: {}, onSelectConversation: { _ in }
                ).frame(width: compact ? SidebarView.compactWidth : SidebarView.columnIdealWidth, height: 640))
                try await stage.waitForIdentifier("Sidebar.Images")
                #expect(!stage.identifiers().contains("Sidebar.Search"))
                #expect(!stage.identifiers().contains("Sidebar.ExperimentalHeader"))
            }
            do {
                let stage = GoldenStage(SidebarView(
                    isCompact: compact, selection: .constant(.search), localSearchEnabled: true,
                    chat: chat, onNewChat: {}, onSelectConversation: { _ in }
                ).frame(width: compact ? SidebarView.compactWidth : SidebarView.columnIdealWidth, height: 640))
                try await stage.waitForIdentifier("Sidebar.Search")
                if !compact {
                    try await stage.waitForIdentifier("Sidebar.ExperimentalHeader")
                    let ids = stage.identifiers()
                    #expect(try #require(ids.firstIndex(of: "Sidebar.ExperimentalHeader")) < #require(ids.firstIndex(of: "Sidebar.Search")))
                }
            }
        }
    }

    @Test("The stored gate guards Search detail, recovery, and production spawn")
    func liveWiring() throws {
        let sources = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Sources/Rapid")
        let content = try String(contentsOf: sources.appendingPathComponent("UI/ContentView.swift"), encoding: .utf8)
        #expect(content.contains("@AppStorage(LocalSearchFeatureConfig.enabledKey)"))
        #expect(content.contains("sectionAfterLocalSearchGateChange(current: section, enabled: state.localSearchEnabled)"))
        #expect(content.contains("if !state.localSearchEnabled { localSearch.disappear() }"))
        #expect(content.contains("case .search:\n            if localSearchEnabled {"))
        let server = try String(contentsOf: sources.appendingPathComponent("Server/ServerManager.swift"), encoding: .utf8)
        #expect(server.contains("localSearchEnabled: LocalSearchFeatureConfig.isEnabled()"))
    }

}
