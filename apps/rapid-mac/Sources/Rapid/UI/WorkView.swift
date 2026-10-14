import AppKit
import SwiftUI

struct WorkView: View {
    let server: ServerManager
    let selectedSection: SidebarSection
    let cachedModels: [ModelEntry]
    let startModel: (String) -> Void
    let startDefaultModel: () -> Void
    @State private var agent = WorkAgent.shared
    @State private var message = ""
    @State private var folder: URL?
    @State private var expanded: Set<String> = []
    @State private var submitting = false
    @AppStorage("Rapid.Work.lastFolder") private var privateLastFolder = ""
    @AppStorage("Rapid.WorkFull.lastFolder") private var fullLastFolder = ""
    private var lastFolder: String {
        get { WorkAgent.fullAccess ? fullLastFolder : privateLastFolder }
        nonmutating set {
            if WorkAgent.fullAccess { fullLastFolder = newValue }
            else { privateLastFolder = newValue }
        }
    }

    private var availableModels: [ModelEntry] {
        Self.availableModels(in: cachedModels, on: MacHardware.detect())
    }

    static func availableModels(in catalog: [ModelEntry], on hardware: MacHardware) -> [ModelEntry] {
        catalog.filter {
            $0.cached && $0.kind == .chat
                && ModelSizing.isAvailable(alias: $0.alias, on: hardware, catalogEntry: $0)
        }
    }

    private var recommendedModel: ModelEntry? {
        Self.recommendedModel(in: cachedModels, on: MacHardware.detect())
    }

    static func recommendedModel(in catalog: [ModelEntry], on hardware: MacHardware) -> ModelEntry? {
        let availableModels = availableModels(in: catalog, on: hardware)
        for alias in ["qwen3.6-35b-4bit", "qwen3.6-27b-4bit"] {
            if let entry = availableModels.first(where: { $0.alias.lowercased() == alias }) {
                return entry
            }
        }
        return nil
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Work").font(.title2.bold())
                    Text(folder?.path ?? "Choose a folder to work in")
                        .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
                Spacer()
                Button("Choose Folder") { chooseFolder() }
                    .disabled(submitting || agent.isBusy || agent.isPreparing)
                Button("New Session") {
                    Task { try? await agent.newSession() }
                }.disabled(!agent.isRunning || submitting || agent.isBusy || agent.isPreparing)
            }
            .padding()
            if WorkAgent.fullAccess {
                Text("Full access mode: internet on, commands run without asking. Don't use private documents here.")
                    .font(.callout.bold())
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
                    .background(.orange.opacity(0.25))
            }
            Divider()
            if server.servingAlias == nil || server.activeBearer == nil {
                ContentUnavailableView {
                    Label("Start a model to use Work", systemImage: "cpu")
                } description: {
                    Text("Work uses the model running in Rapid-MLX Desktop.")
                } actions: {
                    if WorkAgent.fullAccess {
                        if let recommendedModel {
                            Button("Start recommended: \(recommendedModel.alias)") {
                                startModel(recommendedModel.alias)
                            }
                        }
                        Menu("Choose a model") {
                            if availableModels.isEmpty {
                                Text("No cached chat model fits this Mac")
                            } else {
                                ForEach(availableModels, id: \.alias) { entry in
                                    Button(entry.alias) { startModel(entry.alias) }
                                }
                            }
                        }
                    } else {
                        Button("Start Default Model", action: startDefaultModel)
                    }
                }
            } else if folder == nil {
                ContentUnavailableView {
                    Label("Work with local documents", systemImage: "folder")
                } description: {
                    Text("Choose a folder of PDF, CSV, or XLSX files. Ask for a summary or a new file. Your documents stay on this Mac.")
                } actions: {
                    Button("Choose Folder") { chooseFolder() }
                }
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 14) {
                        if !agent.text.isEmpty {
                            Text(agent.text).textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        ForEach(agent.steps) { step in
                            DisclosureGroup(isExpanded: Binding(
                                get: { expanded.contains(step.id) },
                                set: { if $0 { expanded.insert(step.id) } else { expanded.remove(step.id) } }
                            )) {
                                Text(step.command).font(.system(.caption, design: .monospaced))
                                    .textSelection(.enabled)
                                if !step.output.isEmpty {
                                    Text(step.output).font(.system(.caption, design: .monospaced))
                                        .textSelection(.enabled).foregroundStyle(.secondary)
                                }
                            } label: {
                                Label(step.status.capitalized + " · " + step.command.components(separatedBy: "\n")[0],
                                      systemImage: step.status == "completed" ? "checkmark.circle" : "terminal")
                                    .lineLimit(1)
                            }
                            .padding(10).background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 10))
                        }
                        if let error = agent.error {
                            Text(error).foregroundStyle(.red)
                        }
                    }
                    .padding()
                }
                Divider()
                HStack(alignment: .bottom) {
                    TextField("Ask about files, or request a new file…", text: $message, axis: .vertical)
                        .lineLimit(2...5)
                        .onSubmit(send)
                    if agent.isBusy {
                        Button("Stop") { agent.cancel() }
                    } else if submitting || agent.isPreparing {
                        ProgressView("Starting…")
                    } else {
                        Button("Send", action: send)
                            .buttonStyle(.borderedProminent)
                            .disabled(submitting || message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                }.padding()
            }
        }
        .task {
            if ProcessInfo.processInfo.arguments.contains("--work-mvp-synthetic-test") {
                await runSyntheticTest()
                return
            }
            if folder == nil, !lastFolder.isEmpty, FileManager.default.fileExists(atPath: lastFolder) {
                folder = URL(fileURLWithPath: lastFolder)
            } else if folder == nil, WorkAgent.fullAccess {
                let playground = FileManager.default.homeDirectoryForCurrentUser
                    .appendingPathComponent("Desktop/Goose-Playground", isDirectory: true)
                try? FileManager.default.createDirectory(at: playground, withIntermediateDirectories: true)
                folder = playground
            }
            if WorkAgent.fullAccess,
               let probe = ProcessInfo.processInfo.environment["WORK_FULL_LANDING_PROBE"],
               let folder {
                let state = ["bundle": Bundle.main.bundleIdentifier ?? "",
                             "section": String(describing: selectedSection), "folder": folder.path,
                             "model": server.servingAlias ?? ""]
                if let data = try? JSONSerialization.data(withJSONObject: state) {
                    try? data.write(to: URL(fileURLWithPath: probe), options: .atomic)
                }
            }
        }
        .onChange(of: server.activePort) { _, _ in
            if !ProcessInfo.processInfo.arguments.contains("--work-mvp-synthetic-test") { agent.stop() }
        }
        .onChange(of: server.activeBearer) { _, _ in
            if !ProcessInfo.processInfo.arguments.contains("--work-mvp-synthetic-test") { agent.stop() }
        }
        .sheet(item: $agent.permission) { request in
            VStack(alignment: .leading, spacing: 14) {
                Label("Approve this action?", systemImage: "hand.raised")
                    .font(.title3.bold())
                Text("Folder: \(folder?.path ?? "Unknown")").font(.caption)
                ScrollView {
                    Text(request.command)
                        .font(.system(.body, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 300)
                HStack {
                    Spacer()
                    Button("Deny") { agent.answerPermission(request, allow: false) }
                    Button("Allow Once") { agent.answerPermission(request, allow: true) }
                        .buttonStyle(.borderedProminent)
                }
            }
            .padding(24).frame(width: 620)
            .interactiveDismissDisabled()
        }
    }

    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        folder = url
        lastFolder = url.path
        agent.stop()
    }

    private func send() {
        guard let folder, let model = server.servingAlias,
              let bearer = server.activeBearer, !agent.isBusy, !agent.isPreparing, !submitting else { return }
        let prompt = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return }
        submitting = true
        message = ""
        let revision = agent.conversationRevision
        Task {
            defer { submitting = false }
            do {
                guard agent.conversationRevision == revision else { return }
                try await agent.prepare(folder: folder, rapidPort: server.activePort, model: model, bearer: bearer)
                guard agent.conversationRevision == revision else { return }
                await agent.prompt(prompt)
            } catch {
                if agent.conversationRevision == revision { agent.error = error.localizedDescription }
            }
        }
    }

    private func runSyntheticTest() async {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["WORK_MVP_TEST_FOLDER"],
              let model = env["WORK_MVP_TEST_MODEL"],
              let bearer = env["WORK_MVP_TEST_BEARER"],
              let portText = env["WORK_MVP_TEST_PORT"],
              let port = Int(portText) else { return }
        let selected = URL(fileURLWithPath: path)
        folder = selected
        agent.syntheticTestAutoApproval = true
        do {
            try await agent.prepare(folder: selected, rapidPort: port, model: model, bearer: bearer)
            await agent.prompt(env["WORK_MVP_TEST_PROMPT"] ?? "Read fake_w2.pdf, expenses.csv and expenses.xlsx. Write summary.md with W-2 wages and withholding, combined category totals, total expenses and the bookkeeping/tax caveat. Verify every amount with Decimal.")
        } catch {
            agent.error = error.localizedDescription
        }
        let result: [String: Any] = [
            "error": agent.error ?? "",
            "text": agent.text,
            "steps": agent.steps.map { ["command": $0.command, "status": $0.status, "output": $0.output] },
        ]
        if let data = try? JSONSerialization.data(withJSONObject: result) {
            try? data.write(to: selected.appendingPathComponent("work-mvp-test-result.json"), options: .atomic)
        }
    }
}
