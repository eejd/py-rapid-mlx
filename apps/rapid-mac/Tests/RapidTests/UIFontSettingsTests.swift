import Foundation
import SwiftUI
import Testing
@testable import Rapid

@Suite("UI font settings")
struct UIFontSettingsTests {
    private func makeDefaults() -> UserDefaults {
        let name = "UIFontSettingsTests.\(UUID().uuidString)"
        let d = UserDefaults(suiteName: name)!
        d.removePersistentDomain(forName: name)
        return d
    }

    @Test("never returns less than 12pt and scales larger sizes")
    func floorAndScale() {
        let s = UIFontSettings(defaults: makeDefaults())
        #expect(s.scale == 1.0)
        #expect(s.pointSize(9) == 12)
        #expect(s.pointSize(11) == 12)
        #expect(s.pointSize(13) == 13)
        s.scale = 1.5
        #expect(s.pointSize(8) == 12)
        #expect(s.pointSize(10) == 15)
        #expect(s.pointSize(20) == 30)
    }

    @Test("persists the chosen scale")
    func persists() {
        let d = makeDefaults()
        UIFontSettings(defaults: d).scale = 1.3
        #expect(UIFontSettings(defaults: d).scale == 1.3)
    }

    @Test("setScale clamps to 1.0–1.5 and rounds to 0.01; out-of-range saved values fall back")
    func clampAndRound() {
        let d = makeDefaults()
        let s = UIFontSettings(defaults: d)
        s.setScale(2)
        #expect(s.scale == 1.5)
        s.setScale(0.5)
        #expect(s.isDefault)
        s.setScale(1.234)
        #expect(s.scale == 1.23)
        #expect(UIFontScaleControl.label(1.1) == "1.1×")
        d.set(9.0, forKey: UIFontSettings.storageKey)
        #expect(UIFontSettings(defaults: d).scale == 1.0)
    }

    @Test("every text style resolves to at least the floor")
    func textStylesFloored() {
        let s = UIFontSettings(defaults: makeDefaults())
        for style in [Font.TextStyle.caption2, .caption, .footnote, .subheadline, .callout] {
            #expect(s.pointSize(UIFontSettings.basePointSize(style)) >= UIFontSettings.minimumPointSize)
        }
    }
}
