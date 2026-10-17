import Foundation
import Testing

@Suite("SwiftMath default font staging")
struct SwiftMathFontStagingTests {
    @Test("The app stages only the default font pair and all three licences")
    func stagesDefaultFontOnly() throws {
        let result = try Self.runStaging()
        #expect(result.status == 0, "Staging failed: \(result.stderr)")
        #expect(Set(result.files.keys) == Set(Self.requiredFiles))
        for file in Self.requiredFiles {
            #expect(result.files[file] == "fixture: \(file)")
        }
    }

    @Test("Staging fails when a required font resource or licence is missing",
          arguments: Self.requiredFiles)
    func rejectsMissingRequiredFile(file: String) throws {
        let result = try Self.runStaging(missingFile: file)
        #expect(result.status != 0, "Staging succeeded without \(file)")
        #expect(result.stderr.contains("required vendored SwiftMath resource missing:"))
        #expect(result.stderr.contains(file))
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

    private static let requiredFiles = [
        "latinmodern-math.otf", "latinmodern-math.plist",
        "LICENSE", "OFL.txt", "GUST-FONT-LICENSE.txt",
    ]

    private static func runStaging(missingFile: String? = nil) throws -> (
        status: Int32, files: [String: String], stderr: String
    ) {
        let script = try String(
            contentsOf: appRoot.appendingPathComponent("scripts/build.sh"),
            encoding: .utf8
        )
        let start = try #require(script.range(of: "# >>> swiftmath-font-staging\n"))
        let end = try #require(script.range(
            of: "# <<< swiftmath-font-staging", range: start.upperBound..<script.endIndex
        ))
        let block = String(script[start.upperBound..<end.lowerBound])

        let fm = FileManager.default
        let root = URL(fileURLWithPath: "/private/tmp/harbor-slim-app-font-staging")
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? fm.removeItem(at: root) }
        let source = root.appendingPathComponent(
            "Vendor/SwiftMath/Sources/SwiftMath/mathFonts.bundle", isDirectory: true
        )
        let contents = root.appendingPathComponent("Rapid Fixture.app/Contents", isDirectory: true)
        let destination = contents.appendingPathComponent("Resources/mathFonts.bundle")
        try fm.createDirectory(at: source, withIntermediateDirectories: true)
        try fm.createDirectory(at: destination, withIntermediateDirectories: true)

        let fonts = [
            "Asana-Math", "Euler-Math", "FiraMath-Regular", "Garamond-Math",
            "KpMath-Light", "KpMath-Sans", "LeteSansMath", "LibertinusMath-Regular",
            "NotoSansMath-Regular", "latinmodern-math", "texgyretermes-math", "xits-math",
        ]
        let sourceFiles = fonts.flatMap { ["\($0).otf", "\($0).plist"] }
            + ["LICENSE", "OFL.txt", "GUST-FONT-LICENSE.txt"]
        for file in sourceFiles where file != missingFile {
            try "fixture: \(file)".write(
                to: source.appendingPathComponent(file), atomically: true, encoding: .utf8
            )
        }
        try "stale font".write(
            to: destination.appendingPathComponent("stale-extra.otf"),
            atomically: true, encoding: .utf8
        )

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = ["-c", "set -euo pipefail\n" + block]
        var environment = ProcessInfo.processInfo.environment
        environment["ROOT"] = root.path
        environment["CONTENTS"] = contents.path
        process.environment = environment
        let stderr = Pipe()
        process.standardError = stderr
        try process.run()
        let errors = stderr.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()

        var files: [String: String] = [:]
        for file in try fm.contentsOfDirectory(atPath: destination.path) {
            files[file] = try String(
                contentsOf: destination.appendingPathComponent(file), encoding: .utf8
            )
        }
        return (process.terminationStatus, files, String(decoding: errors, as: UTF8.self))
    }

    private static var appRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // RapidTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // rapid-mac
    }
}
