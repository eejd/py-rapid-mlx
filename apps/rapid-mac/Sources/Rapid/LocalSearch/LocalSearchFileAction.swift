import Foundation
import UniformTypeIdentifiers

/// Only known document/media types are eligible for the default viewer.
/// Launchable, indirect, and unknown types always stay in Finder.
enum LocalSearchFileAction {
    enum Action { case open, reveal, unavailable }

    static func classify(type: UTType?, executable: Bool = false,
                         indirect: Bool = false, directory: Bool = false) -> Action {
        guard !executable, !indirect, !directory, let type else { return .reveal }
        let launchable: [UTType] = [.application, .applicationBundle, .executable, .script,
                                    .shellScript, .diskImage, .aliasFile, .symbolicLink, .bundle]
        let launchableIDs = ["com.apple.installer-package", "com.apple.installer-meta-package",
                             "com.apple.shortcuts.workflow", "com.apple.automator-workflow",
                             "com.apple.applescript.script", "com.apple.applescript.script-bundle"]
        if launchable.contains(where: { type.conforms(to: $0) }) || launchableIDs.contains(where: {
            type.identifier == $0 || UTType($0).map { type.conforms(to: $0) } == true
        }) { return .reveal }
        // Include common concrete types too: a restricted process can have
        // type identities without access to the system's ancestor declarations.
        // Unknown identities still fail closed; this is never suffix matching.
        let documents: [UTType] = [.text, .image, .audiovisualContent, .pdf, .rtf,
                                   .spreadsheet, .presentation, .compositeContent,
                                   .plainText, .utf8PlainText, .utf16PlainText,
                                   .png, .jpeg, .heic, .tiff, .gif,
                                   .mpeg4Movie, .quickTimeMovie, .mp3, .wav]
        return documents.contains { type.conforms(to: $0) } ? .open : .reveal
    }

    static func action(for url: URL) -> Action {
        guard url.isFileURL else { return .unavailable }
        // lstat-style attributes identify links without following their target.
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path) else { return .unavailable }
        if attributes[.type] as? FileAttributeType == .typeSymbolicLink { return .reveal }
        let permissions = (attributes[.posixPermissions] as? NSNumber)?.intValue ?? 0
        if permissions & 0o111 != 0 || attributes[.type] as? FileAttributeType == .typeDirectory { return .reveal }
        guard let values = try? url.resourceValues(forKeys: [.contentTypeKey, .isAliasFileKey, .isRegularFileKey, .isDirectoryKey]) else { return .reveal }
        return classify(type: values.contentType, executable: permissions & 0o111 != 0,
                        indirect: values.isAliasFile == true, directory: values.isDirectory == true)
    }
}
