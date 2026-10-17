import Foundation
import Testing

@Suite("Desktop debug-map stripping")
struct DebugMapBuildScriptTests {
    @Test("Only Rapid strips debug symbols before signing")
    func stripsDebugMapsBeforeSigning() throws {
        let script = try String(contentsOf: Self.scriptURL, encoding: .utf8)
        let executable = #""$CONTENTS/MacOS/Rapid""#
        let strip = try #require(script.range(of: "strip -S \(executable)"))
        // Check the first signing invocation in either identity path.
        let sign = try #require(script.range(of: "codesign --force"))
        #expect(strip.lowerBound < sign.lowerBound,
                "Rapid debug-map stripping must precede signing.")
        #expect(!script.contains("strip -x \(executable)"),
                "Local symbols must remain available to crash reports.")
        #expect(!script.contains("strip \(executable)"),
                "Plain strip would remove crash-report symbols.")
        let assertion = try #require(script.range(of: "assert_debug_map_removed \(executable)"))
        #expect(strip.lowerBound < assertion.lowerBound)
        #expect(assertion.lowerBound < sign.lowerBound)

        let rpath = try #require(script.range(of: "install_name_tool -add_rpath"))
        let mainStrip = try #require(script.range(of: #"strip -S "$CONTENTS/MacOS/Rapid""#))
        #expect(rpath.lowerBound < mainStrip.lowerBound)
    }

    @Test("Helper interpreter keeps its symbols unchanged")
    func helperIsNotStripped() throws {
        let script = try String(contentsOf: Self.scriptURL, encoding: .utf8)
        // Pin the sole strip invocation to Rapid; the copied interpreter has no debug map.
        let stripCommands = script.split(separator: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { $0.hasPrefix("if ! strip ") || $0.hasPrefix("strip ") }
        #expect(stripCommands == [#"if ! strip -S "$CONTENTS/MacOS/Rapid"; then"#])
        #expect(script.contains(#"cp "$CONTENTS/Resources/rapid-mlx/python/bin/python3.12" \"#
            + "\n        \"$CUA_HELPER_EXECUTABLE\""))
        #expect(!script.contains(#"assert_debug_map_removed "$CUA_HELPER_EXECUTABLE""#))
    }

    @Test("Local debugging can bypass Rapid's strip block")
    func debugMapRetentionIsOptIn() throws {
        let script = try String(contentsOf: Self.scriptURL, encoding: .utf8)
        let guardText = #"if [[ "${RAPID_KEEP_DEBUG_MAP:-0}" != "1" ]]; then"#
        let guardedStrip = "\(guardText)\n    "
            + #"if ! strip -S "$CONTENTS/MacOS/Rapid"; then"#
        #expect(script.contains(guardedStrip))
    }

    @Test("Post-strip assertion requires text symbols and rejects STABS")
    func verifiesSymbolTable() throws {
        let script = try String(contentsOf: Self.scriptURL, encoding: .utf8)
        #expect(script.contains(#"if ! nm -a "$1" | awk '"#))
        #expect(script.contains("/ T / { has_text = 1 }"))
        #expect(script.contains("/ - / { has_debug = 1 }"))
        #expect(script.contains("END { exit (!has_text || has_debug) }"))
    }

    private static var scriptURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("scripts/build.sh")
    }
}
