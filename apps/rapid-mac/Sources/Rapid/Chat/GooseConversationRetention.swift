import CryptoKit
import Foundation
import SQLite3

/// A helper session belongs to the Desktop conversation that first opened it.
/// The folder is captured because users can move a later turn to another folder.
struct GooseOwnedSession: Codable, Equatable {
    let id: String
    let folder: String
}

enum GooseConversationRetention {
    static func deleteContractsOwned(by conversationID: UUID, folders: [String],
                                     root: URL = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/Rapid/agent-goose-chat/workspaces")) throws {
        for folder in Set(folders) {
            let digest = SHA256.hash(data: Data(folder.utf8))
                .map { String(format: "%02x", $0) }.joined()
            let contract = root.appendingPathComponent(digest).appendingPathComponent("chat-contract.json")
            guard FileManager.default.fileExists(atPath: contract.path) else { continue }
            let safe = try GooseToolPresentation.containedDestination(contract, root: root)
            guard let data = try? Data(contentsOf: safe),
                  let values = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  values["conversation_id"] as? String == conversationID.uuidString else { continue }
            try FileManager.default.removeItem(at: safe)
        }
    }

    /// Goose stores one SQLite database per selected workspace. Its pinned
    /// session manager deletes message and usage rows before the session row.
    static func deleteSessions(_ sessions: [GooseOwnedSession], root: URL = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/Rapid/agent-goose-chat/workspaces")) throws {
        for group in Dictionary(grouping: sessions, by: \.folder) {
            let digest = SHA256.hash(data: Data(group.key.utf8))
                .map { String(format: "%02x", $0) }.joined()
            let database = root.appendingPathComponent(digest)
                .appendingPathComponent("data/sessions/sessions.db")
            guard FileManager.default.fileExists(atPath: database.path) else { continue }
            _ = try GooseToolPresentation.containedDestination(database, root: root)
            var handle: OpaquePointer?
            guard sqlite3_open_v2(database.path, &handle, SQLITE_OPEN_READWRITE, nil) == SQLITE_OK,
                  let handle else {
                if let handle { sqlite3_close(handle) }
                throw CocoaError(.fileReadUnknown)
            }
            defer { sqlite3_close(handle) }
            sqlite3_busy_timeout(handle, 5_000)
            guard sqlite3_exec(handle, "BEGIN IMMEDIATE", nil, nil, nil) == SQLITE_OK else {
                throw CocoaError(.fileWriteUnknown)
            }
            do {
                for session in Set(group.value.map(\.id)) {
                    for table in ["messages", "usage_ledger", "sessions"] {
                        let sql = "DELETE FROM \(table) WHERE \(table == "sessions" ? "id" : "session_id") = ?"
                        var statement: OpaquePointer?
                        guard sqlite3_prepare_v2(handle, sql, -1, &statement, nil) == SQLITE_OK else {
                            throw CocoaError(.fileWriteUnknown)
                        }
                        defer { sqlite3_finalize(statement) }
                        let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
                        guard sqlite3_bind_text(statement, 1, session, -1, transient) == SQLITE_OK,
                              sqlite3_step(statement) == SQLITE_DONE else {
                            throw CocoaError(.fileWriteUnknown)
                        }
                    }
                }
                guard sqlite3_exec(handle, "COMMIT", nil, nil, nil) == SQLITE_OK else {
                    throw CocoaError(.fileWriteUnknown)
                }
            } catch {
                sqlite3_exec(handle, "ROLLBACK", nil, nil, nil)
                throw error
            }
        }
    }

    static func deleteAttachmentDirectories(_ paths: [String], folders: [String],
                                            messages: [ChatMessage], hashes: [String: String] = [:]) throws {
        let manager = FileManager.default
        for path in Set(paths) {
            let directory = URL(fileURLWithPath: path, isDirectory: true)
            guard folders.contains(where: { folder in
                directory.deletingLastPathComponent().standardizedFileURL.path
                    == URL(fileURLWithPath: folder).appendingPathComponent("Attachments").standardizedFileURL.path
            }) else { continue }
            guard manager.fileExists(atPath: directory.path) else { continue }
            let root = directory.deletingLastPathComponent().deletingLastPathComponent()
            _ = try GooseToolPresentation.containedDestination(directory, root: root)
            guard let messageID = UUID(uuidString: directory.lastPathComponent),
                  let message = messages.first(where: { $0.id == messageID }) else { continue }
            let filenames = message.imageAttachments.map {
                $0.id.uuidString + "-" + URL(fileURLWithPath: $0.filename).lastPathComponent
            } + message.imageAttachments.map { URL(fileURLWithPath: $0.filename).lastPathComponent }
                + message.fileAttachments.map { $0.id.uuidString + ".txt" }
                + message.fileAttachments.map {
                    URL(fileURLWithPath: $0.filename).deletingPathExtension().lastPathComponent + ".txt"
                }
            for name in filenames {
                let file = try GooseToolPresentation.containedDestination(
                    directory.appendingPathComponent(name), root: root)
                guard manager.fileExists(atPath: file.path),
                      let data = try? Data(contentsOf: file) else { continue }
                let actualHash = SHA256.hash(data: data)
                    .map { String(format: "%02x", $0) }.joined()
                let legacyData: Data?
                if let image = message.imageAttachments.first(where: {
                    URL(fileURLWithPath: $0.filename).lastPathComponent == name
                }) {
                    legacyData = image.data
                } else if let document = message.fileAttachments.first(where: {
                    URL(fileURLWithPath: $0.filename).deletingPathExtension().lastPathComponent + ".txt" == name
                }) {
                    legacyData = Data(document.extractedText.utf8)
                } else {
                    legacyData = nil
                }
                let legacyHash = legacyData.map {
                    SHA256.hash(data: $0).map { String(format: "%02x", $0) }.joined()
                }
                guard actualHash == hashes[file.path] || actualHash == legacyHash else { continue }
                try manager.removeItem(at: file)
            }
            if try manager.contentsOfDirectory(atPath: directory.path).isEmpty {
                try manager.removeItem(at: directory)
            }
        }
    }
}
