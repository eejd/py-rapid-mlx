import AppKit
import SwiftUI

struct WorkView: View {
    let server: ServerManager
    let startDefaultModel: () -> Void
    @State private var agent = WorkAgent.shared
    @State private var message = ""
    @State private var folder: URL?
    @State private var expanded: Set<String> = []
    @State private var submitting = false
    @AppStorage("Rapid.Work.lastFolder") private var lastFolder = ""

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
                    .disabled(submitting || agent.isBusy)
                Button("New Session") {
                    Task { try? await agent.newSession() }
                }.disabled(!agent.isRunning || submitting || agent.isBusy)
            }
            .padding()
            Divider()
            if server.servingAlias == nil || server.activeBearer == nil {
                ContentUnavailableView {
                    Label("Start a model to use Work", systemImage: "cpu")
                } description: {
                    Text("Work uses the model running in Rapid-MLX Desktop.")
                } actions: {
                    Button("Start Default Model", action: startDefaultModel)
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
                    } else if submitting {
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
            }
        }
        .onChange(of: server.activePort) { _, _ in agent.stop() }
        .onChange(of: server.activeBearer) { _, _ in agent.stop() }
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
              let bearer = server.activeBearer, !agent.isBusy, !submitting else { return }
        let prompt = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return }
        submitting = true
        message = ""
        Task {
            defer { submitting = false }
            do {
                try await agent.prepare(folder: folder, rapidPort: server.activePort, model: model, bearer: bearer)
                await agent.prompt(prompt)
            } catch {
                agent.error = error.localizedDescription
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
            await agent.prompt("Read fake_w2.pdf, expenses.csv and expenses.xlsx. Write summary.md with W-2 wages and withholding, combined category totals, total expenses and the bookkeeping/tax caveat. Verify every amount with Decimal.")
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
