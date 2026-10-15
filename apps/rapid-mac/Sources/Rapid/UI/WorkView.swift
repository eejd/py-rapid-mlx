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
    @State private var pendingExample: String?
    @AppStorage("Rapid.Work.lastFolder") private var privateLastFolder = ""
    @AppStorage("Rapid.WorkFull.lastFolder") private var fullLastFolder = ""
    @AppStorage("Rapid.WorkFull.folderExplicitlyChosen") private var fullFolderExplicitlyChosen = false
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
        let downloaded = catalog.filter { $0.cached && $0.kind == .chat }
        let recommended = recommendedModel(in: downloaded, on: hardware)?.alias
        return downloaded.sorted { left, right in
            if left.alias == recommended { return true }
            if right.alias == recommended { return false }
            return left.alias.localizedStandardCompare(right.alias) == .orderedAscending
        }
    }

    private var recommendedModel: ModelEntry? {
        Self.recommendedModel(in: cachedModels, on: MacHardware.detect())
    }

    static func recommendedModel(in catalog: [ModelEntry], on hardware: MacHardware) -> ModelEntry? {
        let fittingModels = catalog.filter {
            $0.cached && $0.kind == .chat
                && ModelSizing.isAvailable(alias: $0.alias, on: hardware, catalogEntry: $0)
        }
        for alias in ["qwen3.6-27b-4bit", "qwen3.6-35b-4bit", "qwen3.5-9b-4bit"] {
            if let entry = fittingModels.first(where: { $0.alias.lowercased() == alias }) {
                return entry
            }
        }
        return nil
    }

    static func fullFolder(homeDirectory: URL, savedPath: String, explicitlyChosen: Bool,
                           fileManager: FileManager = .default) throws -> URL {
        if explicitlyChosen, !savedPath.isEmpty {
            var isDirectory: ObjCBool = false
            if fileManager.fileExists(atPath: savedPath, isDirectory: &isDirectory), isDirectory.boolValue {
                return URL(fileURLWithPath: savedPath, isDirectory: true)
            }
        }
        // Older Full installs saved a folder without recording whether the user chose it.
        let playground = homeDirectory
            .appendingPathComponent("Desktop", isDirectory: true)
            .appendingPathComponent("Goose-Playground", isDirectory: true)
        try fileManager.createDirectory(at: playground, withIntermediateDirectories: true)
        return playground
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
                    .accessibilityIdentifier("WorkView.ChooseFolder")
                    .disabled(submitting || agent.isBusy || agent.isPreparing || agent.isOpeningSession)
                Button("New Session") {
                    Task { try? await agent.newSession() }
                }.accessibilityIdentifier("WorkView.NewSession")
                    .disabled(!agent.isRunning || submitting || agent.isBusy || agent.isPreparing || agent.isOpeningSession)
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
            if WorkAgent.fullAccess, folder != nil, agent.text.isEmpty, agent.steps.isEmpty {
                fullLanding
            } else if server.servingAlias == nil || server.activeBearer == nil {
                ContentUnavailableView {
                    Label("Start a model to use Work", systemImage: "cpu")
                } description: {
                    Text("Work uses the model running in Rapid-MLX.")
                } actions: {
                    if WorkAgent.fullAccess {
                        if let recommendedModel {
                            Button("Start recommended: \(recommendedModel.alias)") {
                                startModel(recommendedModel.alias)
                            }
                            .accessibilityIdentifier("WorkView.StartRecommendedModel")
                        }
                        Menu("Choose a model") {
                            if availableModels.isEmpty {
                                Text("No downloaded chat models")
                            } else {
                                ForEach(availableModels, id: \.alias) { entry in
                                    Button(entry.alias) { startModel(entry.alias) }
                                        .accessibilityIdentifier("WorkView.StartModel.\(entry.alias)")
                                }
                            }
                        }
                        .accessibilityIdentifier("WorkView.ChooseModel")
                    } else {
                        Button("Start Default Model", action: startDefaultModel)
                            .accessibilityIdentifier("WorkView.StartDefaultModel")
                    }
                }
            } else if folder == nil {
                ContentUnavailableView {
                    Label("Work with local documents", systemImage: "folder")
                } description: {
                    Text("Choose a folder of PDF, CSV, or XLSX files. Ask for a summary or a new file. Your documents stay on this Mac.")
                } actions: {
                    Button("Choose Folder") { chooseFolder() }
                        .accessibilityIdentifier("WorkView.ChooseFolderEmptyState")
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
                            .accessibilityIdentifier("WorkView.Step.\(step.id)")
                            .padding(10).background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 10))
                        }
                        if let error = agent.error {
                            Text(error).foregroundStyle(.red)
                        }
                    }
                    .padding()
                }
                Divider()
                composer
            }
        }
        .task {
            agent.beginFreshFullLanding()
            if ProcessInfo.processInfo.arguments.contains("--work-mvp-synthetic-test") {
                await runSyntheticTest()
                return
            }
            if folder == nil, WorkAgent.fullAccess {
                do {
                    folder = try Self.fullFolder(homeDirectory: FileManager.default.homeDirectoryForCurrentUser,
                                                 savedPath: fullLastFolder,
                                                 explicitlyChosen: fullFolderExplicitlyChosen)
                } catch {
                    agent.error = error.localizedDescription
                }
            } else if folder == nil, !lastFolder.isEmpty, FileManager.default.fileExists(atPath: lastFolder) {
                folder = URL(fileURLWithPath: lastFolder)
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
            sendPendingExampleIfReady()
        }
        .onChange(of: server.servingAlias) { _, _ in sendPendingExampleIfReady() }
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
                        .accessibilityIdentifier("WorkView.PermissionDeny")
                    Button("Allow Once") { agent.answerPermission(request, allow: true) }
                        .accessibilityIdentifier("WorkView.PermissionAllowOnce")
                        .buttonStyle(.borderedProminent)
                }
            }
            .padding(24).frame(width: 620)
            .interactiveDismissDisabled()
        }
    }

    private static let exampleTasks = [
        "Build a snake game as one HTML file",
        "Summarize today's Hacker News posts about local AI into a web page",
        "Write a Python expense tracker with tests",
        "Turn the files in this folder into a tidy README",
    ]

    private var fullLanding: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 24)
            VStack(alignment: .leading, spacing: 20) {
                Text("What should we work on?")
                    .font(.largeTitle.weight(.semibold))
                    .accessibilityIdentifier("WorkView.LandingHeadline")
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                    ForEach(Self.exampleTasks, id: \.self) { task in
                        Button { sendExample(task) } label: {
                            Text(task)
                                .font(.body)
                                .frame(maxWidth: .infinity, minHeight: 62, alignment: .leading)
                                .padding(14)
                                .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .background(.quaternary.opacity(0.45), in: RoundedRectangle(cornerRadius: 12))
                        .accessibilityIdentifier("WorkView.Example.\(Self.exampleTasks.firstIndex(of: task) ?? 0)")
                    }
                }
                if server.servingAlias == nil || server.activeBearer == nil {
                    HStack(spacing: 12) {
                        Text("Work uses the model running in Rapid-MLX.")
                            .foregroundStyle(.secondary)
                        if let recommendedModel {
                            Button("Start recommended: \(recommendedModel.alias)") {
                                startModel(recommendedModel.alias)
                            }
                            .accessibilityIdentifier("WorkView.StartRecommendedModel")
                        }
                        Menu("Choose a model") {
                            if availableModels.isEmpty {
                                Text("No downloaded chat models")
                            } else {
                                ForEach(availableModels, id: \.alias) { entry in
                                    Button(modelMenuTitle(entry)) { startModel(entry.alias) }
                                        .accessibilityIdentifier("WorkView.StartModel.\(entry.alias)")
                                }
                            }
                        }
                        .accessibilityIdentifier("WorkView.ChooseModel")
                    }
                }
                if let error = agent.error {
                    Text(error).foregroundStyle(.red)
                        .accessibilityIdentifier("WorkView.Error")
                }
            }
            .frame(maxWidth: 760)
            .padding(24)
            Spacer(minLength: 24)
            composer
        }
    }

    private func modelMenuTitle(_ entry: ModelEntry) -> String {
        let status = entry.alias == recommendedModel?.alias ? "Recommended · downloaded" : "Downloaded"
        return [entry.alias, entry.sizeOnDisk, status].compactMap { $0 }.joined(separator: " · ")
    }

    private var canSend: Bool {
        folder != nil && server.servingAlias != nil && server.activeBearer != nil
            && !submitting && !agent.isPreparing && !agent.isOpeningSession
            && !message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var sendDisabledReason: String? {
        if folder == nil { return "Choose a folder to start" }
        if server.servingAlias == nil || server.activeBearer == nil { return "Load a model to start" }
        if message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return "Enter a task to send" }
        return nil
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(alignment: .bottom) {
                if WorkAgent.fullAccess {
                    WorkPromptField(text: $message, onSubmit: send)
                        .accessibilityIdentifier("WorkView.Message")
                        .frame(minHeight: 48, maxHeight: 100)
                } else {
                    TextField("Ask about files, or request a new file…", text: $message, axis: .vertical)
                        .accessibilityIdentifier("WorkView.Message")
                        .lineLimit(2...5)
                        .onSubmit(send)
                }
                if agent.isBusy {
                    Button("Stop") { agent.cancel() }
                        .accessibilityIdentifier("WorkView.Stop")
                } else if submitting || agent.isPreparing || agent.isOpeningSession {
                    ProgressView("Starting…")
                } else {
                    Button("Send", action: send)
                        .accessibilityIdentifier("WorkView.Send")
                        .buttonStyle(.borderedProminent)
                        .disabled(!canSend)
                    if !canSend, let sendDisabledReason {
                        Text(sendDisabledReason)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .accessibilityIdentifier("WorkView.SendDisabledReason")
                    }
                }
            }
            if WorkAgent.fullAccess {
                Text("Press Enter to send")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
    }

    private func sendExample(_ task: String) {
        message = task
        if server.servingAlias != nil, server.activeBearer != nil {
            send()
        } else if let recommendedModel {
            pendingExample = task
            startModel(recommendedModel.alias)
        }
    }

    private func sendPendingExampleIfReady() {
        guard let pendingExample, server.servingAlias != nil, server.activeBearer != nil else { return }
        self.pendingExample = nil
        message = pendingExample
        send()
    }

    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        folder = url
        lastFolder = url.path
        if WorkAgent.fullAccess { fullFolderExplicitlyChosen = true }
        agent.stop()
    }

    private func send() {
        guard let folder, let model = server.servingAlias,
              let bearer = server.activeBearer, !agent.isBusy, !agent.isPreparing,
              !agent.isOpeningSession, !submitting else { return }
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

/// NSTextView forwards accessibility value changes to the same binding as typing.
/// SwiftUI's TextField may update its AX value without committing its text binding.
struct WorkPromptField: NSViewRepresentable {
    @Binding var text: String
    let onSubmit: () -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.borderType = .bezelBorder
        let editor = AccessibleEditor(frame: NSRect(x: 0, y: 0, width: 480, height: 48))
        editor.isRichText = false
        editor.isAutomaticQuoteSubstitutionEnabled = false
        editor.isAutomaticDashSubstitutionEnabled = false
        editor.isVerticallyResizable = true
        editor.isHorizontallyResizable = false
        editor.autoresizingMask = [.width]
        editor.textContainer?.widthTracksTextView = true
        editor.font = .preferredFont(forTextStyle: .body)
        editor.setAccessibilityLabel("Message to Work")
        editor.setAccessibilityIdentifier("WorkView.Message")
        editor.textContainerInset = NSSize(width: 8, height: 8)
        editor.delegate = context.coordinator
        editor.valueChanged = { context.coordinator.parent.text = $0 }
        scroll.documentView = editor
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        context.coordinator.parent = self
        guard let editor = scroll.documentView as? AccessibleEditor else { return }
        if editor.string != text { editor.string = text }
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: WorkPromptField
        init(_ parent: WorkPromptField) { self.parent = parent }
        func textDidChange(_ notification: Notification) {
            guard let editor = notification.object as? NSTextView else { return }
            parent.text = editor.string
        }
        func textView(_ textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
            guard commandSelector == #selector(NSResponder.insertNewline(_:)) else { return false }
            parent.onSubmit()
            return true
        }
    }

    final class AccessibleEditor: NSTextView {
        var valueChanged: ((String) -> Void)?
        override func setAccessibilityValue(_ value: Any?) {
            super.setAccessibilityValue(value)
            if let value = value as? String {
                string = value
                valueChanged?(value)
            }
        }
    }
}
