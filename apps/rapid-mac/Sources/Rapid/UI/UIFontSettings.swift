import Foundation
import Observation
import SwiftUI

/// App-wide UI text size (Settings → Appearance → Text size).
///
/// Every chrome font goes through ``Font/rapid(size:weight:design:)``,
/// which multiplies by ``scale`` and never returns less than
/// ``minimumPointSize``. Because this is `@Observable` and the font
/// helpers read ``scale`` during `body`, SwiftUI's observation tracking
/// re-renders every view that used a scaled font when the user changes
/// the setting — no relaunch, no view identity reset.
///
/// Not `@MainActor`: fonts are also built from nonisolated statics
/// (``RapidFont``). Writes only happen from the Settings UI on main.
@Observable
final class UIFontSettings: @unchecked Sendable {
    static let shared = UIFontSettings()

    static let storageKey = "rapid.uiFontSettings.scale.v1"
    /// Floor for every UI font — anything smaller is unreadable for many users.
    static let minimumPointSize: CGFloat = 12
    static let defaultScale: Double = 1.0
    /// Settings picker choices (multiplier, label).
    static let choices: [(scale: Double, label: String)] = [
        (1.0, String(localized: "Default")),
        (1.15, String(localized: "Large")),
        (1.3, String(localized: "Larger")),
        (1.5, String(localized: "Largest")),
    ]

    @ObservationIgnored private let defaults: UserDefaults

    var scale: Double {
        didSet { defaults.set(scale, forKey: Self.storageKey) }
    }

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        let saved = defaults.double(forKey: Self.storageKey)
        self.scale = saved > 0 ? saved : Self.defaultScale
    }

    func pointSize(_ base: CGFloat) -> CGFloat {
        max(Self.minimumPointSize, (base * scale).rounded())
    }

    /// macOS point sizes for the SwiftUI text styles.
    static func basePointSize(_ style: Font.TextStyle) -> CGFloat {
        switch style {
        case .largeTitle: 26
        case .title: 22
        case .title2: 17
        case .title3: 15
        case .headline, .body: 13
        case .callout: 12
        case .subheadline: 11
        default: 10 // footnote, caption, caption2
        }
    }
}

extension Font {
    /// Scaled, floored replacement for `Font.system(size:weight:design:)`.
    static func rapid(
        size: CGFloat,
        weight: Font.Weight = .regular,
        design: Font.Design = .default
    ) -> Font {
        .system(size: UIFontSettings.shared.pointSize(size), weight: weight, design: design)
    }

    /// Scaled, floored replacement for `Font.system(_:design:weight:)` / `.caption` etc.
    static func rapid(
        _ style: Font.TextStyle,
        design: Font.Design = .default,
        weight: Font.Weight? = nil
    ) -> Font {
        .rapid(
            size: UIFontSettings.basePointSize(style),
            weight: weight ?? (style == .headline ? .semibold : .regular),
            design: design
        )
    }
}
