import Foundation
import Testing

@Suite("SwiftMath default font staging")
struct SwiftMathFontStagingTests {
    @Test("The app stages only the default font pair and all three licences")
    func stagesDefaultFontOnly() throws {
        let script = try String(
            contentsOf: Self.appRoot.appendingPathComponent("scripts/build.sh"),
            encoding: .utf8
        )

        #expect(script.contains(#"SWIFTMATH_DEFAULT_FONT="latinmodern-math""#))
        #expect(script.contains(#"""
        SWIFTMATH_REQUIRED_FILES=(
            "${SWIFTMATH_DEFAULT_FONT}.otf"
            "${SWIFTMATH_DEFAULT_FONT}.plist"
            "LICENSE"
            "OFL.txt"
            "GUST-FONT-LICENSE.txt"
        )
        """#))
        #expect(script.contains(#"mkdir -p "$CONTENTS/Resources/mathFonts.bundle""#))
        #expect(script.contains(#"""
        for file in "${SWIFTMATH_REQUIRED_FILES[@]}"; do
            if [[ ! -f "$SWIFTMATH_FONTS/$file" ]]; then
                echo "ERR: required vendored SwiftMath resource missing: $SWIFTMATH_FONTS/$file" >&2
                exit 1
            fi
            cp "$SWIFTMATH_FONTS/$file" "$CONTENTS/Resources/mathFonts.bundle/"
        done
        """#))
        #expect(script.contains(#"""
        if [[ -f "$SWIFTMATH_FONTS/Info.plist" ]]; then
            cp "$SWIFTMATH_FONTS/Info.plist" "$CONTENTS/Resources/mathFonts.bundle/"
        fi
        """#))
        #expect(!script.contains(#"cp -R "$SWIFTMATH_FONTS""#),
                "Copying the whole bundle would ship unused math fonts again.")
    }

    @Test("The vendored default still uses the staged Latin Modern font")
    func vendorDefaultIsLatinModern() throws {
        let source = try String(
            contentsOf: Self.appRoot.appendingPathComponent(
                "Vendor/SwiftMath/Sources/SwiftMath/MathRender/MTFontManager.swift"
            ),
            encoding: .utf8
        )
        #expect(source.contains("""
            public var defaultFont: MTFont? {
                MTFontManager.fontManager.latinModernFont(withSize: kDefaultFontSize)
            }
        """), "A changed vendored default requires updating the app's staged font.")
        #expect(source.contains("""
            public func latinModernFont(withSize size:CGFloat) -> MTFont? {
                MTFontManager.fontManager.font(withName: "latinmodern-math", size: size)
            }
        """))
    }

    private static var appRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // RapidTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // rapid-mac
    }
}
