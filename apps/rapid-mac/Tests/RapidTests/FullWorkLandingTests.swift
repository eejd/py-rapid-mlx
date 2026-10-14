import Foundation
import Testing
@testable import Rapid

@MainActor
@Suite("Full Work landing")
struct FullWorkLandingTests {
    private let fullID = "com.rapidmlx.rapid.work-full"

    @Test("Full opens Work while the private flavor keeps Chat")
    func initialSection() {
        #expect(SidebarSection.initial(bundleIdentifier: fullID, environment: [:]) == .work)
        #expect(SidebarSection.initial(bundleIdentifier: "com.rapidmlx.rapid.work-mvp", environment: [:]) == .chat)
        #expect(SidebarSection.initial(bundleIdentifier: fullID, environment: [
            "RAPID_GUI_GOLDEN_MODE": "1", "RAPID_GUI_INITIAL_SECTION": "images",
        ]) == .images)
    }

    @Test("Work recommends the largest cached Qwen that fits RAM")
    func cachedRecommendation() {
        let catalog = [
            ModelEntry(alias: "qwen3.6-35b-4bit", hfRepo: nil, sizeOnDisk: nil, cached: true),
            ModelEntry(alias: "qwen3.6-27b-4bit", hfRepo: nil, sizeOnDisk: nil, cached: true),
        ]
        let large = MacHardware(brandString: "M3 Max", family: .m3, tier: .max,
                                physicalRAMBytes: 64 << 30, memoryBandwidthGBs: 400)
        let small = MacHardware(brandString: "M3 Pro", family: .m3, tier: .pro,
                                physicalRAMBytes: 16 << 30, memoryBandwidthGBs: 150)
        #expect(WorkView.recommendedModel(in: catalog, on: large)?.alias == "qwen3.6-35b-4bit")
        #expect(WorkView.recommendedModel(in: catalog, on: small) == nil)
        #expect(WorkView.recommendedModel(in: catalog.filter { $0.alias.contains("27b") },
                                          on: large)?.alias == "qwen3.6-27b-4bit")
    }

    @Test("Full uses and creates the Playground until its own folder chooser is used")
    func fullFolderSelection() throws {
        let fileManager = FileManager.default
        let home = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? fileManager.removeItem(at: home) }
        let desktop = home.appendingPathComponent("Desktop", isDirectory: true)
        let playground = desktop.appendingPathComponent("Goose-Playground", isDirectory: true)

        for stalePath in ["", home.path, desktop.path] {
            let folder = try WorkView.fullFolder(homeDirectory: home, savedPath: stalePath,
                                                 explicitlyChosen: false, fileManager: fileManager)
            #expect(folder == playground)
            var isDirectory: ObjCBool = false
            #expect(fileManager.fileExists(atPath: folder.path, isDirectory: &isDirectory))
            #expect(isDirectory.boolValue)
        }

        let chosen = home.appendingPathComponent("Chosen", isDirectory: true)
        try fileManager.createDirectory(at: chosen, withIntermediateDirectories: true)
        #expect(try WorkView.fullFolder(homeDirectory: home, savedPath: chosen.path,
                                        explicitlyChosen: true, fileManager: fileManager) == chosen)
    }
}
