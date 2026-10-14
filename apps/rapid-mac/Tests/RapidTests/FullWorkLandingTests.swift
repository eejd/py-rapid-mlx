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
}
