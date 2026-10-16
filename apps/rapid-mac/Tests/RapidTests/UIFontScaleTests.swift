import Foundation
import SwiftUI
import Testing
@testable import Rapid

@Suite("UI font scale")
struct UIFontScaleTests {
    private func makeDefaults() -> UserDefaults {
        let name = "UIFontScaleTests.\(UUID().uuidString)"
        let d = UserDefaults(suiteName: name)!
        d.removePersistentDomain(forName: name)
        return d
    }

    @Test("never returns less than 12pt and scales larger sizes")
    func floorAndScale() {
        let s = UIFontScale(defaults: makeDefaults())
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
        UIFontScale(defaults: d).scale = 1.3
        #expect(UIFontScale(defaults: d).scale == 1.3)
    }

    @Test("every text style resolves to at least the floor")
    func textStylesFloored() {
        let s = UIFontScale(defaults: makeDefaults())
        for style in [Font.TextStyle.caption2, .caption, .footnote, .subheadline, .callout] {
            #expect(s.pointSize(UIFontScale.basePointSize(style)) >= UIFontScale.minimumPointSize)
        }
    }
}
