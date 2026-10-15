import AppKit
import Foundation
import Observation

/// Per-window controller that ties ``SessionStore`` to a
/// ``ChatStreamClient``. The view owns a ``ChatViewModel`` and the model
/// owns the active streaming ``Task`` — that makes "stop" a single
/// ``task.cancel()`` instead of state-machine bookkeeping.
///
@MainActor
@Observable
final class ChatViewModel {
    /// The ACTIVE conversation's live message buffer. All streaming /
    /// append / update goes through here; ``persistActive()`` snapshots it
    /// into ``conversations`` + disk at each turn boundary (M3 history).
    ///
    /// This is the visible root-to-leaf PATH, not the whole tree. Branching
    /// deliberately did not change that: every streaming write addresses a
    /// message by its index in this array, so keeping it linear left all of
    /// that untouched. The turns the user has branched away from live in
    /// ``branchedAway`` and are re-joined at persist time.
    private(set) var messages: [ChatMessage] = []

    /// Nodes of the active conversation's tree that are NOT on the visible
    /// path — the answers a Regenerate replaced, the prompts an edit
    /// superseded, and everything that hung below them.
    ///
    /// Held apart from ``messages`` purely so the live buffer stays a plain
    /// linear array (see above). The two are concatenated back into one bag
    /// of nodes by ``persistActive`` and split apart again on load, so
    /// ``ChatConversation.messages`` remains the single serialised tree.
    private var branchedAway: [ChatMessage] = []

    /// Parent → last-chosen-child for the active conversation's forks. Kept
    /// alongside the split buffers and persisted with them; see
    /// ``ChatConversation/branchChoices`` for why it exists.
    private var branchChoices: [UUID: UUID] = [:]

    /// Monotonic stamp of the tree's SHAPE — bumped when nodes are added,
    /// moved between the two buffers, or removed. Streamed tokens landing in
    /// an existing row leave it alone.
    ///
    /// Exists for ``treeSnapshot``: the per-row queries (`branchPosition`,
    /// `deletionImpact`, `siblings`) run inside the transcript's ForEach, so
    /// they re-execute for every visible row on every render pass — including
    /// the pass after each streamed token. Rescanning and re-sorting
    /// `branchedAway + messages` there is quadratic in transcript length at
    /// thirty invalidations a second; reading a snapshot rebuilt only on
    /// shape changes is not.
    private var treeShapeVersion = 0

    /// The whole tree, indexed: rebuilt lazily whenever ``treeShapeVersion``
    /// moves. ``children`` groups are pre-sorted in sibling order.
    ///
    /// READ-ONLY by contract. Because content writes do not move the shape
    /// version, a snapshot built while an answer streams holds that row's
    /// EARLY value; only the fields that never mutate after append — id,
    /// role, parentID, createdAt — may be trusted from it. Every structural
    /// mutation (``selectBranch``, ``deleteMessage``) re-reads node VALUES
    /// from the live buffers via ``liveTree()`` before adopting or
    /// persisting; feeding ``nodes`` back into ``adoptTree`` would restore a
    /// truncated mid-stream answer over the finished one.
    private struct TreeSnapshot {
        var version = -1
        var nodes: [ChatMessage] = []
        var byID: [UUID: ChatMessage] = [:]
        var children: [UUID: [ChatMessage]] = [:]
        var roots: [ChatMessage] = []

        func childGroup(of parentID: UUID?) -> [ChatMessage] {
            guard let parentID else { return roots }
            return children[parentID] ?? []
        }
    }

    private var cachedTree = TreeSnapshot()

    /// The current tree snapshot, rebuilding it first if the shape moved.
    private func treeSnapshot() -> TreeSnapshot {
        if cachedTree.version == treeShapeVersion { return cachedTree }
        // Path first: if a corrupt store ever produces the same id in both
        // buffers, the visible row is the one that wins everywhere.
        let nodes = MessageTree.deduplicatingByID(messages + branchedAway)
        var snapshot = TreeSnapshot()
        snapshot.version = treeShapeVersion
        snapshot.nodes = nodes
        snapshot.byID.reserveCapacity(nodes.count)
        for node in nodes { snapshot.byID[node.id] = node }
        for node in nodes {
            if let parent = node.parentID {
                snapshot.children[parent, default: []].append(node)
            } else {
                snapshot.roots.append(node)
            }
        }
        for key in snapshot.children.keys {
            snapshot.children[key]?.sort(by: MessageTree.precedes)
        }
        snapshot.roots.sort(by: MessageTree.precedes)
        cachedTree = snapshot
        return snapshot
    }

    /// The whole tree with CURRENT node values, straight from the live
    /// buffers. The input to every structural mutation — never the cached
    /// snapshot, whose values may predate the tokens that streamed in since
    /// it was built.
    private func liveTree() -> [ChatMessage] {
        MessageTree.deduplicatingByID(messages + branchedAway)
    }

    /// Saved conversations, newest-updated first — the sidebar "Older"
    /// list. Loaded from disk on init; upserted whenever the active
    /// conversation gains a user turn.
    private(set) var conversations: [ChatConversation] = []

    /// User-created folders for filing conversations. Loaded from their own
    /// `folders.json` on init; see ``ConversationFolderStore`` for why the
    /// list is not stored alongside the transcripts.
    private(set) var folders: [ChatFolder] = []

    /// Identity of the conversation ``messages`` currently holds. A fresh
    /// UUID on launch (opens to an empty "Ask anything"); ``persistActive``
    /// upserts under this id once the user sends.
    private(set) var activeConversationID = UUID()
    /// Set only by the local New Chat command. Restore/import/sync paths never
    /// populate it, so UI preferences cannot infer consent from a list delta.
    private(set) var locallyCreatedConversationID: UUID?

    /// User-authored instructions for the open conversation. They are kept
    /// outside the visible transcript and merged into the wire-only system row.
    private(set) var conversationInstructions: String = ""

    /// Bumped on every conversation switch (new / select / delete). Each
    /// send captures the epoch; any streaming write or completion whose
    /// captured epoch no longer matches is discarded — so a stream that
    /// outlives a switch can't bleed tokens into, or persist over, the
    /// conversation the user moved to (codex BLOCKING).
    private var conversationEpoch = 0
    private var gooseConversationID: UUID?
    private var gooseSessionID: String?
    private var gooseContextIDs: [UUID] = []
    private var goosePermissions: [String: (WorkPermission, String)] = [:]
    private var gooseFolder: URL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Rapid-MLX Workspace", isDirectory: true)
    private(set) var gooseFallback: String?
    var gooseFolderPath: String { gooseFolder.path }

    func setGooseFolder(_ url: URL) { gooseFolder = url.standardizedFileURL }
    func retryGoose() { gooseFallback = nil; WorkAgent.shared.stop(); gooseConversationID = nil }
    func activateGooseFallback(_ error: Error) {
        gooseFallback = String(localized: "Tools are unavailable. Chat is using the direct model. Retry")
        WorkAgent.shared.stop()
    }

    func answerGoosePermission(_ request: WorkPermission, tool: String, allow: Bool, always: Bool = false) {
        if always && allow { GoosePermissionPolicy.alwaysAllow(tool) }
        WorkAgent.shared.answerPermission(request, allow: allow)
        if let index = messages.lastIndex(where: { $0.gooseParts?.contains(where: { $0.tool?.id == request.toolCallID }) == true }),
           let part = messages[index].gooseParts?.firstIndex(where: { $0.tool?.id == request.toolCallID }) {
            messages[index].gooseParts?[part].tool?.permissionState = allow ? (always ? "always allowed" : "allowed") : "declined"
        }
    }
    func goosePermission(for id: String) -> (WorkPermission, String)? { goosePermissions[id] }
    func allowPendingGoosePermissions() {
        for (request, tool) in goosePermissions.values {
            if messages.contains(where: { $0.gooseParts?.contains(where: { $0.tool?.id == request.toolCallID && $0.tool?.permissionState == "permission needed" }) == true }) {
                answerGoosePermission(request, tool: tool, allow: true)
            }
        }
    }

    /// ``var`` not ``let`` because ``ChatStreamClient`` is a struct
    /// and ``send()`` re-targets ``client.baseURL`` to track
    /// ``ServerManager.activePort`` (v0.5.6 port-fallback) before each
    /// request. A class wrapper would have been an alternative but
    /// would force every test that constructs a model to either
    /// inject a real URLSession or thread a mock through both layers.
    private var client: ChatStreamClient

    /// Set while a stream is in flight. UI reads this to show the stop
    /// button instead of send.
    /// The assistant message currently streaming, as (id, text).
    ///
    /// A projection rather than a stored property: it exists so exactly one
    /// view can observe the growing string, instead of every row observing
    /// the whole `messages` array. Nil once nothing is streaming.
    var streamingBody: StreamingBody? {
        guard isStreaming,
              let m = messages.last(where: { $0.role == .assistant && $0.status == .streaming })
        else { return nil }
        return StreamingBody(id: m.id, text: m.content)
    }

    struct StreamingBody: Equatable {
        let id: UUID
        let text: String
    }

    private(set) var isStreaming: Bool = false {
        didSet {
            // A turn just ended (stream finished, failed, or was stopped) —
            // snapshot the final exchange into history + disk. Covers every
            // completion path without hooking each one.
            if oldValue && !isStreaming {
                persistActive()
                scheduleBackgroundAssist()
                scheduleMemoryExtraction()
            }
        }
    }

    /// The model alias used for the most recent turn, captured at
    /// ``beginAssistantTurn`` so the background memory extractor can
    /// address the same engine after the stream finishes.
    private var lastTurnAlias: String?
    /// Most recent transport / parse error. Cleared on the next ``send``.
    /// Shown as a banner above the compose bar so the user knows what
    /// went wrong without scraping the log tail.
    private(set) var lastError: String?
    /// Structured counterpart to ``lastError`` used by the shared failure
    /// view to choose a recovery action without parsing display copy.
    private(set) var lastFailureKind: FailureDiagnosis.Kind?
    /// Which model ``lastError`` is about.
    ///
    /// Without this the readiness surface had to guess, and it guessed
    /// "whatever is selected right now" — so after `kimi-k2.6` failed to
    /// start, choosing `bonsai-1.7b-2bit` re-labelled Kimi's error as
    /// Bonsai's. A failure belongs to the model that actually failed;
    /// recording the alias is what lets a later surface decide whether
    /// the failure is still the user's current problem.
    ///
    /// Cleared in lockstep with ``lastError``. Note this scopes only the
    /// BANNER — the failed turn stays in the transcript either way.
    private(set) var lastFailureAlias: String?

    private var inflight: Task<Void, Never>?

    /// The title / follow-up completions. One handle for both arms, so one
    /// `cancel()` stops everything this model started on its own account.
    private var backgroundAssist: Task<Void, Never>?

    /// Chips under the last answer. Never persisted — a restored transcript
    /// shows no suggestions, which is right: they were about a moment.
    private(set) var followUp: FollowUpState = .idle

    /// The message the current chips belong to. Publishing is gated on this
    /// still being the last row, which covers a new turn, an edit, a retry,
    /// and a conversation switch in one condition.
    private(set) var followUpAnchorID: UUID?

    enum FollowUpState: Equatable {
        /// Nothing asked for, or nothing came back.
        case idle
        /// Asked; the rail is holding its space so the arrival costs no layout.
        case pending
        case ready([String])
    }

    /// Tests that exercise turn replay can disable disk I/O so seeded
    /// transcripts never read or overwrite the user's conversation history.
    /// Production keeps the default enabled.
    private let persistsConversations: Bool
    private let conversationStoreURL: URL?

    /// v0.4.14: user-mutable sampling knobs. Optional in the init
    /// signature so existing tests don't have to spin one up — they
    /// fall back to the v0.4.12 hard-coded defaults via
    /// ``ChatStreamClient.Request``'s own default parameters.
    /// Production code (``RapidApp.init``) always passes a real
    /// ``SamplingConfig`` reading from ``UserDefaults``.
    let sampling: SamplingConfig?

    /// Global custom instructions shared with Settings.
    let customInstructions: CustomInstructionsConfig

    /// Persistent memory entries. Injected into the system prompt and
    /// updated by the background extractor after each completed turn.
    /// Optional so unit tests can construct a viewmodel without a real
    /// memory store; production wires this from ``RapidApp.init``.
    let memoryStore: MemoryStore?

    /// v0.5.1: live handle on the embedded server so the chat loop can
    /// resolve `request.model` against the alias currently being served
    /// instead of trusting the picker bar's state. Mirrors the global
    /// "loaded model" model used by Ollama and LM Studio — outgoing
    /// requests follow whatever the backend has resident. Optional so
    /// unit tests can construct a viewmodel without spinning up a real
    /// process; production wires this from ``RapidApp.init``.
    private weak var server: ServerManager?

    /// App-owned lifecycle signal for a real, visible assistant completion.
    /// Kept as a callback so chat has no dependency on telemetry policy.
    private let onProductValueDelivered: @MainActor (ProductValueKind) -> Void
    /// Injectable so conversation deletion can remove extracts in isolated tests.
    private let documentCache: DocumentContentCache

    init(
        client: ChatStreamClient = ChatStreamClient(),
        sampling: SamplingConfig? = nil,
        customInstructions: CustomInstructionsConfig? = nil,
        memoryStore: MemoryStore? = nil,
        server: ServerManager? = nil,
        persistsConversations: Bool = true,
        conversationStoreURL: URL? = nil,
        onProductValueDelivered: @escaping @MainActor (ProductValueKind) -> Void = { _ in },
        documentCache: DocumentContentCache = .shared
    ) {
        self.client = client
        self.sampling = sampling
        self.customInstructions = customInstructions ?? CustomInstructionsConfig()
        self.memoryStore = memoryStore
        self.server = server
        self.persistsConversations = persistsConversations
        self.conversationStoreURL = conversationStoreURL
        self.onProductValueDelivered = onProductValueDelivered
        self.documentCache = documentCache
        let loadedConversations = persistsConversations
            ? ConversationStore.load(from: conversationStoreURL)
            : []
        self.conversations = WorkAgent.gooseChat
            ? Self.repairGooseTitles(loadedConversations) : loadedConversations
        self.folders = persistsConversations
            ? ConversationFolderStore.load(from: Self.folderStoreURL(for: conversationStoreURL))
            : []
    }

    /// The folder file that sits beside whichever conversation store is in
    /// use, so an injected test store keeps its folders in the same temp
    /// directory instead of touching the real one.
    private static func folderStoreURL(for conversationStore: URL?) -> URL? {
        ConversationFolderStore.companionURL(forConversationStore: conversationStore)
    }

    // MARK: - Conversation history (M3)

    static func repairGooseTitles(_ conversations: [ChatConversation]) -> [ChatConversation] {
        conversations.map { original in
            var conversation = original
            conversation.messages = conversation.messages.map(repairGooseError)
            conversation.branches = conversation.branches.map(repairGooseError)
            guard !conversation.hasCustomTitle,
                  conversation.messages.contains(where: { $0.role == .user }) else { return conversation }
            conversation.title = ConversationStore.title(from: conversation.messages)
            conversation.hasGeneratedTitle = false
            return conversation
        }
    }

    private static func repairGooseError(_ original: ChatMessage) -> ChatMessage {
        var message = original
        guard message.role == .assistant else { return message }
        if message.content.contains("Ran into this error: Server error"),
           message.content.contains("127.0.0.1") {
            message.gooseErrorDetails = message.content
            clearGooseFailedText(&message)
            message.errorMessage = gooseFailureCopy
            message.status = .failed
        } else if message.status == .failed, let raw = message.errorMessage,
                  raw.contains("127.0.0.1") || raw.contains("Bad Gateway") {
            message.gooseErrorDetails = raw
            message.errorMessage = gooseFailureCopy
        }
        return message
    }

    /// Snapshot the active conversation into ``conversations`` + disk. A
    /// no-op until the user has actually sent something (no empty rows in
    /// the sidebar). Title is derived from the first user message.
    ///
    /// - Parameter touching: whether this counts as *activity*. Only a real
    ///   change to the transcript should refresh ``updatedAt`` and bubble the
    ///   row to the top. Merely opening a conversation to read it, or leaving
    ///   it to start a new one, archives the buffer without claiming the user
    ///   did anything to it — otherwise clicking through history silently
    ///   reshuffles the sidebar, and the list stops reflecting when each
    ///   conversation was last *worked on*.
    private func persistActive(touching: Bool = true) {
        guard messages.contains(where: { $0.role == .user }) else { return }
        let now = Date()
        // Title comes from the VISIBLE path: the sidebar row must match the
        // conversation the user sees, not a prompt they branched away from.
        let title = ConversationStore.title(from: messages)
        // The visible path and the branches it was split from go to SEPARATE
        // fields. `messages` stays a linear transcript so a downgrade renders
        // the conversation the user was looking at instead of a flattened pile
        // of alternatives; `branches` is a new key an old build ignores.
        //
        // The leaf pointer is only meaningful once alternatives exist — for a
        // linear conversation the decoder's `messages.last` fallback IS the
        // leaf. Omitting it keeps every branching key absent from a
        // conversation that never branched.
        let leaf = branchedAway.isEmpty ? nil : messages.last?.id
        if conversations.contains(where: { $0.id == activeConversationID }) {
            conversations = ConversationOrdering.updating(
                conversations,
                id: activeConversationID,
                touching: touching,
                at: now
            ) { conversation in
                conversation.messages = messages
                conversation.branches = branchedAway
                conversation.activeLeafID = leaf
                conversation.branchChoices = branchChoices
                // A renamed row keeps its name. Re-deriving unconditionally
                // meant the next streamed token silently reverted the user's
                // rename back to the first prompt's opening words.
                // Two owners can take the title away from the derivation:
                // the user (a rename) and the background titler. Neither may
                // be reverted by the next streamed token.
                if !conversation.hasCustomTitle && !conversation.hasGeneratedTitle {
                    conversation.title = title
                }
                conversation.customInstructions = Self.normalizedInstruction(
                    conversationInstructions
                )
                conversation.gooseSessionID = gooseSessionID
                conversation.gooseFolder = gooseFolder.path
            }
        } else {
            conversations.insert(
                ChatConversation(
                    id: activeConversationID,
                    title: title,
                    messages: messages,
                    branches: branchedAway,
                    activeLeafID: leaf,
                    branchChoices: branchChoices,
                    createdAt: now,
                    updatedAt: now,
                    customInstructions: Self.normalizedInstruction(conversationInstructions),
                    gooseSessionID: gooseSessionID,
                    gooseFolder: gooseFolder.path
                ),
                at: 0
            )
        }
        saveConversations()
    }

    // MARK: - Background assist (titles, follow-ups)

    /// Every task this model started on its own account, plus the reader's
    /// stream.
    ///
    /// Folded into one helper because all five call sites were already
    /// performing the identical two-line dance for ``inflight``, and a second
    /// task would have made that six places to remember. The class of bug
    /// this removes is "somebody added a third task and missed site four".
    private func cancelInflightWork() {
        if WorkAgent.gooseChat && isStreaming { WorkAgent.shared.stop() }
        inflight?.cancel()
        inflight = nil
        backgroundAssist?.cancel()
        backgroundAssist = nil
        // The rail belongs to the assist, so it is torn down with it. Keeping
        // this here rather than at the call sites is the same argument the
        // helper was created for, and the same bug: of the six sites, three
        // remembered to put the rail away and three did not. A cancelled
        // assist exits without publishing, so nothing downstream clears the
        // anchor — and `ChatView` mounts the rail on the anchor alone, at a
        // fixed height whatever it shows.
        clearFollowUps()
    }

    /// A settled assistant answer worth building on.
    struct SettledTurn: Equatable {
        let assistantID: UUID
        let lastUserText: String
        /// The answer being followed up on. Used as the language reference,
        /// because it is longer and more representative of the conversation
        /// than a user message that might be one English word.
        let assistantText: String
    }

    /// The transcript ends in an answer a reader could act on, or nil.
    ///
    /// Mirrors ``MessageRow``'s `showsAssistantActions`: an answer worth a
    /// Copy button is an answer worth following up on, and keeping the two
    /// judgements the same means chips never appear under a row that offers
    /// no other affordance. Adds one clause of its own — a turn the reader
    /// stopped is one they did not want more of.
    nonisolated static func settledTurn(_ messages: [ChatMessage]) -> SettledTurn? {
        guard let last = messages.last,
              last.role == .assistant,
              last.status == .complete,
              // ``finaliseCancellation`` marks a stopped turn this way.
              last.errorMessage != "Stopped.",
              !last.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              // A tool-dispatch shell is scaffolding, not an answer.
              (last.toolCalls?.isEmpty ?? true),
              let user = messages.last(where: { $0.role == .user })
        else { return nil }
        return SettledTurn(
            assistantID: last.id, lastUserText: user.content, assistantText: last.content
        )
    }

    /// Is this the first exchange — the only one that gets a title?
    nonisolated static func isFirstExchange(_ messages: [ChatMessage]) -> Bool {
        messages.filter { $0.role == .user }.count == 1
    }

    /// Whether the engine's lane will batch our request alongside the
    /// reader's, or serialise it in front of their next turn.
    ///
    /// The `--mllm` vision lane runs one request at a time, so a background
    /// call there is not free — it is a queue. The live profile's
    /// `serving_lane` is the authoritative execution decision. Its
    /// `modality` remains `text` even for vision-capable models and therefore
    /// cannot distinguish the serialised lane.
    ///
    /// An unknown or older profile allows. Otherwise a sidecar predating
    /// `serving_lane` would disable both features permanently, and the damage
    /// on that path is already bounded by `max_tokens`, the deadline, and the
    /// fact that the reader's next send cancels us. A mismatched profile also
    /// allows rather than letting stale metadata decide for another alias.
    nonisolated static func laneAllowsBackgroundWork(
        _ profile: ServerModelProfile?, alias: String
    ) -> Bool {
        guard let profile,
              profile.id.caseInsensitiveCompare(alias) == .orderedSame,
              let servingLane = profile.servingLane
        else { return true }
        return servingLane == "text"
    }

    /// Ask the model to name this conversation and to propose what to ask
    /// next. Runs at the turn boundary; every reason not to is checked here,
    /// on the main actor, before anything is spawned.
    private func scheduleBackgroundAssist() {
        // Last turn's chips are stale the moment this turn ends, whatever
        // happens next.
        followUp = .idle
        followUpAnchorID = nil

        guard let server, case .ready(let alias) = server.state else { return }
        guard Self.laneAllowsBackgroundWork(server.activeModelProfile, alias: alias) else { return }
        guard let turn = Self.settledTurn(messages) else { return }

        let conversationID = activeConversationID
        let epoch = conversationEpoch
        let transcript = messages
        // Not "is this the first exchange": that forfeits the one chance
        // permanently if the call is cancelled, which `send` does on the very
        // next message. `hasGeneratedTitle` already means "a title has
        // landed", so gating on it keeps the once-per-conversation promise
        // while letting a cancelled or failed attempt be retried next turn.
        // Goose Chat history uses the first user message as its stable title.
        // The generic background titler would replace it after the reply.
        let needsTitle = !WorkAgent.gooseChat && conversations.first { $0.id == conversationID }
            .map { !$0.hasCustomTitle && !$0.hasGeneratedTitle } == true

        followUp = .pending
        followUpAnchorID = turn.assistantID
        // Alias, endpoint, and credential are one lifecycle snapshot. Reading
        // port/bearer later from inside the task could pair a stale alias with
        // a replacement server after a model switch.
        let scheduledTarget = BackgroundCompletionClient.Target(
            port: server.activePort,
            bearer: server.activeBearer,
            alias: alias
        )

        backgroundAssist = Task { [weak self] in
            guard let self else { return }
            guard let target = Self.revalidatedBackgroundTarget(
                scheduledTarget,
                state: server.state,
                activePort: server.activePort,
                activeBearer: server.activeBearer
            ) else {
                self.clearFollowUps(anchoredTo: turn.assistantID)
                return
            }
            let client = BackgroundCompletionClient()

            // Run both requests concurrently, but publish each one as soon as
            // its own deadline completes. The title may wait for 15 seconds;
            // it must not hold an already-ready follow-up rail hostage.
            let titlePrompt = needsTitle
                ? ConversationTitleSuggestion.messages(forFirstExchange: transcript)
                : nil
            let followUpPrompt = FollowUpSuggestion.messages(forTurn: transcript)
            await Self.deliverBackgroundReplies(
                title: {
                    guard let titlePrompt else { return nil }
                    return await client.complete(
                        titlePrompt, target: target, maxTokens: 24,
                        temperature: 0.2, deadline: .seconds(15)
                    )
                },
                followUp: {
                    guard let followUpPrompt else { return nil }
                    return await client.complete(
                        followUpPrompt, target: target, maxTokens: 96,
                        temperature: 0.6, topP: 0.95, deadline: .seconds(8)
                    )
                },
                onFollowUp: { suggestions in
                    guard !Task.isCancelled, epoch == self.conversationEpoch else {
                        // Cancelled after this assist reserved its space.
                        // Clear only the rail generation this task owns.
                        self.clearFollowUps(anchoredTo: turn.assistantID)
                        return false
                    }
                    self.publishFollowUps(
                        suggestions, anchoredTo: turn.assistantID,
                        excluding: turn.lastUserText, reference: turn.assistantText
                    )
                    return true
                },
                onTitle: { title in
                    guard !Task.isCancelled, epoch == self.conversationEpoch else { return }
                    if let title { self.applyGeneratedTitle(title, to: conversationID) }
                }
            )
        }
    }

    /// Start both optional completions together, while letting the
    /// turn-scoped follow-up result reach the UI before the longer title
    /// deadline. Returning `false` after the follow-up cancels the structured
    /// title child when the owning conversation or turn is already stale.
    static func deliverBackgroundReplies(
        title: @escaping @Sendable () async -> String?,
        followUp: @escaping @Sendable () async -> String?,
        onFollowUp: (String?) -> Bool,
        onTitle: (String?) -> Void
    ) async {
        async let titleReply = title()
        async let followUpReply = followUp()

        let suggestions = await followUpReply
        guard onFollowUp(suggestions) else { return }

        let generatedTitle = await titleReply
        onTitle(generatedTitle)
    }

    /// Keep a background request bound to the server generation that
    /// scheduled it.
    ///
    /// The task starts cooperatively, so a stop, restart, or model switch can
    /// occur after ``scheduleBackgroundAssist`` captures `.ready` but before
    /// this check executes. Returning the original snapshot only when all
    /// lifecycle-facing fields still agree prevents a stale alias from being
    /// sent to a replacement endpoint.
    static func revalidatedBackgroundTarget(
        _ scheduled: BackgroundCompletionClient.Target,
        state: ServerState,
        activePort: Int,
        activeBearer: String?
    ) -> BackgroundCompletionClient.Target? {
        guard state == .ready(alias: scheduled.alias),
              activePort == scheduled.port,
              activeBearer == scheduled.bearer
        else { return nil }
        return scheduled
    }

    /// Show the chips, if they are still about the answer on screen.
    ///
    /// Split out rather than left inline so the staleness rule is reachable
    /// from a test without a server — the same reason
    /// ``finishStartupCancellation`` is its own method.
    func publishFollowUps(
        _ raw: String?,
        anchoredTo anchorID: UUID,
        excluding lastUserText: String,
        reference: String? = nil
    ) {
        // One condition covers a switched conversation, a deleted one, a new
        // turn, and an edit or retry that rewrote the tail.
        guard messages.last?.id == anchorID else {
            clearFollowUps()
            return
        }
        guard let raw else { clearFollowUps(); return }
        let questions = FollowUpSuggestion.parse(
            raw, excluding: lastUserText, reference: reference
        )
        guard !questions.isEmpty else { clearFollowUps(); return }
        // Sets the anchor as well as the state. `scheduleBackgroundAssist`
        // already set it before spawning, but publishing one without the other
        // is a combination the screen can never show — ``ChatView`` mounts the
        // rail on the anchor — so the two are written together here rather
        // than depending on a caller having done half the job.
        followUpAnchorID = anchorID
        followUp = .ready(questions)
    }

    /// Put the rail away.
    ///
    /// Clearing the anchor as well as the state is the load-bearing half:
    /// ``ChatView`` mounts the rail on the anchor alone, and the rail holds a
    /// fixed 40pt whatever it is showing. Leaving the anchor set after a
    /// failure — a deadline, unparseable output, a script mismatch, all of
    /// which are common — left an empty band under the answer until the next
    /// turn. Shrinking the document is safe; it is growth the scroll probe
    /// cannot absorb.
    func clearFollowUps() {
        followUp = .idle
        followUpAnchorID = nil
    }

    /// Clear a rail only while it still belongs to the finishing assist.
    ///
    /// Cancellation is cooperative. A superseded request may resume after a
    /// newer turn has already installed its own rail, so task-local cleanup
    /// must compare ownership rather than clearing shared presentation state
    /// unconditionally.
    func clearFollowUps(anchoredTo anchorID: UUID) {
        guard followUpAnchorID == anchorID else { return }
        clearFollowUps()
    }

    /// Land a machine-generated title. Silent on every rejection.
    ///
    /// Addresses the conversation by id rather than by "whichever is active",
    /// so a late reply cannot rename the wrong row — the epoch check above
    /// already dropped it, and this makes the write safe even if that check
    /// were ever relaxed.
    ///
    /// Deliberately does not touch `updatedAt`: naming a row is not working
    /// on it, so the sidebar must not reshuffle. Same choice
    /// ``renameConversation`` makes, and for the same reason.
    func applyGeneratedTitle(_ raw: String, to id: UUID) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        guard !conversations[index].hasCustomTitle else { return }
        guard !conversations[index].hasGeneratedTitle else { return }
        guard let title = ConversationTitleSuggestion.parse(raw) else { return }
        conversations[index].title = title
        conversations[index].hasGeneratedTitle = true
        saveConversations()
    }

    // MARK: - Conversation row actions (rename / pin / archive)

    /// Rename a saved conversation.
    ///
    /// Trimmed, and blank input is rejected rather than accepted as an empty
    /// row label — an unnamed row already has a sensible derived title, so
    /// "clear the name" is better served by the caller not committing.
    /// Setting ``hasCustomTitle`` is what stops ``persistActive`` re-deriving
    /// the title from the transcript on the next save.
    ///
    /// A rename is not conversation *activity*, so ``updatedAt`` and the row's
    /// position are left alone — the same reasoning ``ConversationOrdering``
    /// applies to merely opening a conversation.
    @discardableResult
    func renameConversation(_ id: UUID, to newTitle: String) -> Bool {
        let trimmed = newTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return false }
        conversations[index].title = trimmed
        conversations[index].hasCustomTitle = true
        saveConversations()
        return true
    }

    /// Pin / unpin a conversation. Pinned rows get their own sidebar section
    /// above the date buckets.
    func setConversationPinned(_ id: UUID, _ pinned: Bool) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        guard conversations[index].isPinned != pinned else { return }
        conversations[index].isPinned = pinned
        // Pinning something archived is contradictory — the row would be
        // pinned to a list it isn't in. Surfacing it is the intent.
        if pinned { conversations[index].isArchived = false }
        saveConversations()
    }

    /// Archive / unarchive a conversation. Archiving is deliberately NOT a
    /// delete: the transcript stays on disk and the row remains reachable
    /// from the sidebar's Archived section, which is why — unlike Delete —
    /// it needs no confirmation and is one click to undo.
    ///
    /// Archiving the OPEN conversation leaves it open. Closing the transcript
    /// out from under the user would turn a filing action into an unexpected
    /// navigation, and re-reading what you just archived is normal.
    func setConversationArchived(_ id: UUID, _ archived: Bool) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        guard conversations[index].isArchived != archived else { return }
        conversations[index].isArchived = archived
        if archived { conversations[index].isPinned = false }
        saveConversations()
    }

    /// Update the instruction layer for the open conversation. Existing saved
    /// chats are written immediately; a brand-new empty chat is persisted with
    /// its first user turn, avoiding an empty row in the sidebar.
    func setConversationInstructions(_ value: String) {
        guard value != conversationInstructions else { return }
        conversationInstructions = value
        guard let index = conversations.firstIndex(where: { $0.id == activeConversationID }) else {
            return
        }
        conversations[index].customInstructions = Self.normalizedInstruction(value)
        saveConversations()
    }

    private func saveConversations() {
        guard persistsConversations else { return }
        ConversationStore.save(conversations, to: conversationStoreURL)
    }

    // MARK: - Folders

    /// Create a folder. Returns nil for a blank name rather than making an
    /// unnamed row the user then can't tell apart from any other.
    ///
    @discardableResult
    func createFolder(named rawName: String) -> ChatFolder? {
        guard let name = ChatFolder.normalizedName(rawName) else { return nil }
        guard !folderNameExists(name) else { return nil }
        let folder = ChatFolder(name: name)
        folders.append(folder)
        saveFolders()
        return folder
    }

    @discardableResult
    func renameFolder(_ id: UUID, to rawName: String) -> Bool {
        guard let name = ChatFolder.normalizedName(rawName) else { return false }
        guard let index = folders.firstIndex(where: { $0.id == id }) else { return false }
        guard folders[index].name != name else { return true }
        guard !folderNameExists(name, excluding: id) else { return false }
        folders[index].name = name
        saveFolders()
        return true
    }

    /// Delete a folder WITHOUT deleting the conversations filed in it.
    ///
    /// The transcripts are the valuable thing; the folder is just where the
    /// user put them. Unfiling them returns the rows to the date buckets,
    /// which is recoverable — deleting them would not be. Same restraint
    /// ``setConversationArchived`` shows for the same reason.
    ///
    /// Clearing ``folderID`` eagerly (rather than leaning on the render-time
    /// orphan fallback) is what keeps a later folder created with a recycled
    /// id from silently adopting rows that were never filed into it.
    func deleteFolder(_ id: UUID) {
        guard folders.contains(where: { $0.id == id }) else { return }
        folders.removeAll { $0.id == id }
        var unfiled = false
        for index in conversations.indices where conversations[index].folderID == id {
            conversations[index].folderID = nil
            unfiled = true
        }
        saveFolders()
        if unfiled { saveConversations() }
    }

    /// File a conversation into a folder, or pass nil to unfile it.
    ///
    /// Filing is not conversation *activity*: ``updatedAt`` and the row's
    /// position are left alone, matching rename / pin / archive and the
    /// contract ``ConversationOrdering`` states.
    ///
    /// **Filing un-archives.** The sidebar shows archived rows only in the
    /// Archived disclosure, ahead of any folder, so filing one without this
    /// would record the folder and change nothing you can see — the row stays
    /// where it was and the action reads as broken. Putting something in a
    /// folder means wanting it in that folder; surfacing it is what makes the
    /// gesture honest. Unfiling (`nil`) deliberately does NOT re-archive:
    /// there is no earlier state to restore, and silently archiving a row the
    /// user only wanted out of a folder would hide it entirely.
    func moveConversation(_ id: UUID, toFolder folderID: UUID?) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        // An id for a folder that no longer exists would file the row into a
        // section that never renders — i.e. it would look deleted.
        if let folderID, !folders.contains(where: { $0.id == folderID }) { return }
        let needsSurfacing = folderID != nil && conversations[index].isArchived
        guard conversations[index].folderID != folderID || needsSurfacing else { return }
        conversations[index].folderID = folderID
        if needsSurfacing { conversations[index].isArchived = false }
        saveConversations()
    }

    private func saveFolders() {
        guard persistsConversations else { return }
        ConversationFolderStore.save(
            folders,
            to: Self.folderStoreURL(for: conversationStoreURL)
        )
    }

    func folderNameExists(_ name: String, excluding excludedID: UUID? = nil) -> Bool {
        folders.contains {
            $0.id != excludedID
                && $0.name.compare(name, options: [.caseInsensitive, .diacriticInsensitive])
                    == .orderedSame
        }
    }

    /// Load a saved conversation into the transcript, archiving whatever is
    /// currently open first. Cancels any in-flight stream.
    func selectConversation(_ id: UUID) {
        guard id != activeConversationID else { return }
        locallyCreatedConversationID = nil
        cancelInflightWork()
        conversationEpoch &+= 1
        // Archive + unstick BEFORE swapping buffers, so the old transcript
        // is what gets persisted and a mid-stream switch doesn't leave the
        // incoming conversation showing Stop.
        isStreaming = false
        persistActive(touching: false)
        guard let conv = conversations.first(where: { $0.id == id }) else { return }
        // Seed BEFORE adopting — the stored map is what tells ``adoptTree``
        // where the user was inside each branch of the incoming conversation.
        branchChoices = conv.branchChoices
        adoptTree(conv.allMessages, activeLeafID: conv.activeLeafID ?? conv.messages.last?.id)
        conversationInstructions = conv.customInstructions ?? ""
        gooseSessionID = conv.gooseSessionID
        gooseFolder = URL(fileURLWithPath: conv.gooseFolder ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Rapid-MLX Workspace").path)
        activeConversationID = id
        lastError = nil
        lastFailureKind = nil
        lastFailureAlias = nil
    }

    /// Deletes a conversation and all document extracts in its full branch tree.
    func deleteConversation(_ id: UUID) {
        var attachmentIDs: Set<UUID> = []
        if id == activeConversationID {
            attachmentIDs.formUnion(liveTree().flatMap { $0.fileAttachments.map(\.id) })
        }
        if let stored = conversations.first(where: { $0.id == id }) {
            attachmentIDs.formUnion(stored.allMessages.flatMap { $0.fileAttachments.map(\.id) })
        }

        // If deleting the OPEN conversation, tear down the live transcript
        // FIRST — otherwise the `isStreaming = false` below fires
        // persistActive() via didSet while the deleted messages + id are
        // still active, re-inserting the conversation we just removed.
        if id == activeConversationID {
            cancelInflightWork()
            conversationEpoch &+= 1
            messages.removeAll()
            branchedAway.removeAll()
            treeShapeVersion += 1
            branchChoices.removeAll()
            conversationInstructions = ""
            activeConversationID = UUID()
            isStreaming = false          // messages now empty → persistActive no-ops
            lastError = nil
            lastFailureKind = nil
            lastFailureAlias = nil
        }
        conversations.removeAll { $0.id == id }
        saveConversations()
        documentCache.remove(contentsOf: attachmentIDs)
    }

    // MARK: - In-memory message storage

    /// Load a stored tree into the split live representation: the visible
    /// path into ``messages``, everything else into ``branchedAway``.
    ///
    /// The inverse of the re-join in ``persistActive``. Splitting on load and
    /// merging on save is what lets the streaming path keep addressing
    /// messages by index into a plain linear array while the conversation as
    /// a whole is still a tree.
    private func adoptTree(_ tree: [ChatMessage], activeLeafID: UUID?) {
        let path = MessageTree.activePath(
            in: tree,
            activeLeafID: activeLeafID,
            preferring: branchChoices
        )
        let visible = Set(path.map(\.id))
        messages = path
        branchedAway = tree.filter { !visible.contains($0.id) }
        treeShapeVersion += 1
        // Record which way this path went at every fork it crosses. MERGED,
        // not replaced: forks that are not on this path keep the position
        // they had, which is the whole point — stepping to a sibling must not
        // wipe out where the user was inside the branch they just left.
        branchChoices.merge(MessageTree.choices(along: path)) { _, new in new }
    }

    /// Append a message and return its index in ``messages``.
    @discardableResult
    private func appendMessage(_ message: ChatMessage) -> Int {
        // Every append extends the visible path, so the new node's parent is
        // whatever currently ends it. Setting the link here — at the single
        // funnel every send / stream / tool-result append already goes
        // through — is what keeps the tree connected without touching any of
        // those call sites.
        var linked = message
        if linked.parentID == nil {
            linked.parentID = messages.last?.id
        }
        messages.append(linked)
        treeShapeVersion += 1
        return messages.count - 1
    }

    /// Overwrite the message at ``index`` when it is still in range.
    private func updateMessage(at index: Int, with message: ChatMessage) {
        guard messages.indices.contains(index) else { return }
        // Streamed tokens replace a row's CONTENT, which the branch metadata
        // snapshot does not care about. Only a write that moves the row to a
        // different node — new id or new parent — is a shape change; today no
        // caller does that, and this guard keeps the invariant honest if one
        // ever starts to.
        if messages[index].id != message.id || messages[index].parentID != message.parentID {
            treeShapeVersion += 1
        }
        messages[index] = message
    }

    /// Streaming write guarded on the sending conversation's epoch: a
    /// no-op once the user has switched conversations under the stream, so
    /// a stale/cancelled stream can't overwrite a message in — or, via the
    /// completion path, persist — the conversation now on screen.
    private func writeStreamMessage(at index: Int, epoch: Int, _ message: ChatMessage) {
        guard epoch == conversationEpoch else { return }
        updateMessage(at: index, with: message)
    }

    /// Snapshot of the message at ``index``, or ``nil`` when out of range.
    private func currentMessage(index: Int) -> ChatMessage? {
        guard messages.indices.contains(index) else { return nil }
        return messages[index]
    }

    /// Persist attachment delivery progress on the user turn itself. Failed
    /// empty assistant placeholders are intentionally removed from later wire
    /// history, so they cannot safely own this focus state or retry budget.
    private func applyImageDeliveryEvent(
        messageID: UUID?,
        event: ImageDeliveryEvent,
        epoch: Int
    ) {
        guard epoch == conversationEpoch else { return }
        Self.applyImageDeliveryEvent(
            in: &messages,
            messageID: messageID,
            event: event
        )
    }

    enum ImageDeliveryEvent: Sendable {
        case accepted
        case terminalRejection
        case transientFailure
        case abandoned
    }

    /// Pure state machine for the request lifecycle. Matching by persisted
    /// message identity prevents a late response from changing a newer image
    /// turn after conversation branching or editing. A first pre-token failure
    /// remains eligible for one implicit retry; the second quarantines that
    /// image. Cancellation or startup failure does not spend the retry budget.
    static func applyImageDeliveryEvent(
        in messages: inout [ChatMessage],
        messageID: UUID?,
        event: ImageDeliveryEvent
    ) {
        guard let messageID,
              let index = messages.firstIndex(where: { $0.id == messageID })
        else { return }

        switch event {
        case .accepted:
            // An explicit Retry may prove a previously rejected image works.
            messages[index].imageDeliveryStatus = .accepted
        case .terminalRejection:
            // Once the server emits a token, a later transport failure means
            // the response was interrupted, not that the image was rejected.
            guard messages[index].imageDeliveryStatus != .accepted else { return }
            messages[index].imageDeliveryStatus = .rejected
        case .transientFailure:
            switch messages[index].imageDeliveryStatus {
            case .accepted, .rejected:
                return
            case .retryable:
                messages[index].imageDeliveryStatus = .rejected
            case .pending, nil:
                messages[index].imageDeliveryStatus = .retryable
            }
        case .abandoned:
            // Only a newly-created request owns `.pending`. A cancelled retry
            // preserves its prior retryable/rejected state by identity.
            if messages[index].imageDeliveryStatus == .pending {
                messages[index].imageDeliveryStatus = nil
            }
        }
    }

    /// Start a fresh conversation — drops the transcript and any stale
    /// error banner. The in-flight stream (if any) is cancelled first.
    func newConversation() {
        cancelInflightWork()
        if WorkAgent.gooseChat {
            WorkAgent.shared.beginNewChat()
            Task { try? await WorkAgent.shared.newSession() }
        }
        conversationEpoch &+= 1
        // Fix: without this a New Chat during a stream leaves the empty
        // chat stuck showing Stop (isStreaming never reset). Setting it
        // false here also archives the just-closed conversation via didSet.
        isStreaming = false
        persistActive(touching: false)
        messages.removeAll()
        branchedAway.removeAll()
        treeShapeVersion += 1
        branchChoices.removeAll()
        conversationInstructions = ""
        activeConversationID = UUID()
        gooseSessionID = nil
        locallyCreatedConversationID = activeConversationID
        if WorkAgent.gooseChat { WorkAgent.shared.preopenGooseChatSession() }
        lastError = nil
        lastFailureKind = nil
        lastFailureAlias = nil
    }

    /// DEV-ONLY: replace the transcript with a fixed set of messages so
    /// the `DevSnapshot` harness can render a populated chat. Never called
    /// in normal use (only from the env-gated snapshot path).
    func devSeedMessages(_ seeded: [ChatMessage]) {
        // Chain the fixture into a tree. Callers hand over a plain linear
        // array, which in tree terms is a set of parentless roots — the
        // transcript would render as a single turn, and a Regenerate on it
        // would branch from nothing. Same repair the ``ChatConversation``
        // initialiser applies, for the same reason.
        messages = MessageTree.repairingLegacyChain(seeded)
        branchedAway.removeAll()
        treeShapeVersion += 1
        branchChoices.removeAll()
    }

    /// Append a locally authored assistant message to the open conversation.
    ///
    /// The one caller is the onboarding completion transaction, which lands
    /// its welcome message in the chat the user is about to be dropped into.
    /// There is no network round trip and no stream: the text is written
    /// straight into the transcript as a finished assistant turn, so it can
    /// never wedge the typing indicator or the streaming gate.
    ///
    /// A transcript that already holds messages is left alone. Onboarding
    /// completion is a one-shot event, but the app can reach it with a live
    /// conversation on screen (a user who skipped setup, chatted, then came
    /// back to it), and injecting a stray intro into the middle of somebody's
    /// conversation is worse than skipping the pleasantry.
    ///
    /// - Returns: ``true`` when the message landed in the transcript.
    @discardableResult
    func seedAssistantWelcome(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        guard messages.isEmpty else { return false }
        appendMessage(ChatMessage(
            role: .assistant,
            content: trimmed,
            wireVisibility: .transcriptOnly
        ))
        return true
    }

    /// Fires a detached background task to review the completed exchange
    /// and upsert any durable facts into ``memoryStore``. Called from the
    /// ``isStreaming`` didSet when a turn ends; never blocks the UI or
    /// competes with an active stream because it runs on a separate task.
    private func scheduleMemoryExtraction() {
        guard let memoryStore, memoryStore.isEnabled else { return }
        guard let alias = lastTurnAlias else { return }
        guard messages.count >= 2 else { return }
        let conversationID = activeConversationID
        let turnMessages: [(role: String, content: String)] = messages
            .filter { $0.role == .user || $0.role == .assistant }
            .map { (role: $0.role.rawValue, content: $0.content) }
        let baseURL = client.baseURL
        let bearer = server?.activeBearer

        Task { [weak self] in
            // The network call hops off MainActor via URLSession's async
            // implementation; we only return to the main actor for the
            // store mutation below.
            guard !Task.isCancelled else { return }
            let extractor = MemoryExtractor(
                baseURL: baseURL,
                bearerToken: bearer
            )
            do {
                let operations = try await extractor.extract(
                    model: alias,
                    messages: turnMessages
                )
                for operation in operations {
                    switch operation {
                    case .add(let content):
                        memoryStore.upsert(content: content, conversationID: conversationID)
                    case .remove(let content):
                        let matching = memoryStore.entries.first {
                            $0.content.localizedCaseInsensitiveContains(content)
                                || content.localizedCaseInsensitiveContains($0.content)
                        }
                        if let match = matching {
                            memoryStore.remove(id: match.id)
                        }
                    }
                }
            } catch {
                // Memory extraction is best-effort; a failure here is
                // invisible to the user and logged only if diagnostics
                // capture it. The next completed turn gets another chance.
            }
        }
    }

    /// Append the user message, open a placeholder assistant row, and
    /// kick off the streaming task. The text field clears immediately on
    /// the caller's side.
    func send(
        _ text: String,
        alias: String,
        supportsImageInput: Bool? = nil,
        imageAttachments: [ChatImageAttachment] = [],
        fileAttachments: [ChatFileAttachment] = []
    ) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty || !imageAttachments.isEmpty || !fileAttachments.isEmpty else { return }
        guard !isStreaming else { return }

        // The reader has taken the wheel. Anything we were asking the model
        // on our own account stops now — it would land on a transcript that
        // has moved on, and on the serialised vision lane it would be sitting
        // in front of the turn they just asked for.
        cancelInflightWork()

        let user = ChatMessage(
            role: .user,
            content: trimmed,
            imageAttachments: imageAttachments,
            imageDeliveryStatus: imageAttachments.isEmpty ? nil : .pending,
            fileAttachments: ChatFileAttachment.fittedForMessage(fileAttachments),
            status: .complete
        )
        _ = appendMessage(user)
        // Surface the conversation in the sidebar immediately — before the
        // (possibly cold, ~minute) model load — so "Older" reflects the
        // send the instant it happens, not only once the reply lands.
        persistActive()

        let resolvedImageCapability = supportsImageInput
            ?? ModelBrandStyle.supportsImageInput(forAlias: alias)
        beginAssistantTurn(
            alias: alias,
            supportsImageInput: resolvedImageCapability,
            imageMessageID: imageAttachments.isEmpty ? nil : user.id
        )
    }

    /// Test seam for lifecycle assertions that need the current turn to be
    /// fully drained. Awaiting the owned task is deterministic under runner
    /// load; polling ``isStreaming`` with a wall-clock deadline is not.
    func _testingWaitForCurrentTurn() async {
        await inflight?.value
    }

    /// Open a streaming assistant turn under whatever currently ends the
    /// visible path, and drive it to completion.
    ///
    /// Split out of ``send`` so a regeneration can answer the user turn ALREADY
    /// in the transcript instead of appending a copy of it. Getting that wrong
    /// branches at the wrong level: the replacement answer would hang under a
    /// duplicate prompt, making it a sibling of the original *question* rather
    /// than of the original *answer*, and the `‹ n/m ›` control would never
    /// see the two answers as alternatives at all.
    private func beginAssistantTurn(
        alias: String,
        supportsImageInput: Bool,
        imageMessageID: UUID? = nil
    ) {
        let placeholder = ChatMessage(role: .assistant, status: .streaming)
        let placeholderIndex = appendMessage(placeholder)

        lastError = nil
        lastFailureKind = nil
        lastFailureAlias = nil
        lastTurnAlias = alias
        isStreaming = true

        let epoch = conversationEpoch
        // Freeze both user-authored layers for this turn. Changing Settings in
        // another window takes effect on the next send, not halfway through a
        // multi-round tool exchange.
        let globalInstruction = customInstructions.global
        let chatInstruction = conversationInstructions
        let memoryContext = memoryStore?.formattedForPrompt()
        inflight = Task { [weak self] in
            guard let self else { return }

            // Bring the model up if it isn't serving yet. The user's
            // turn is already in the transcript, so a load that takes a
            // minute — or fails — costs them nothing they typed.
            // `ensureServing` short-circuits when we are already serving
            // this alias, so the warm path pays only a state read.
            if let server {
                let ready = await server.ensureServing(
                    alias: alias,
                    hfPath: startupHFPath,
                    estimatedMemoryGB: nil,
                    replacementGroup: .assistant
                )
                // A user Stop during the (possibly cold, multi-second)
                // bring-up cancels THIS task. That is a deliberate
                // cancel, not a start failure — route it through the
                // cancel contract, never the "Couldn't start" failure
                // row + error banner. Checked BEFORE `ready`: a Stop
                // that lands the instant the model reaches ``.ready``
                // returns `ready == true`, yet must still not stream and
                // must still reset `isStreaming`.
                guard !Task.isCancelled else {
                    finishStartupCancellation(
                        placeholderIndex: placeholderIndex,
                        imageMessageID: imageMessageID,
                        epoch: epoch
                    )
                    return
                }
                guard ready else {
                    finishWithStartupFailure(
                        placeholderIndex: placeholderIndex,
                        alias: alias,
                        imageMessageID: imageMessageID,
                        epoch: epoch
                    )
                    return
                }
            }
            guard !Task.isCancelled else {
                finishStartupCancellation(
                    placeholderIndex: placeholderIndex,
                    imageMessageID: imageMessageID,
                    epoch: epoch
                )
                return
            }

            // v0.5.6: re-target the chat client at
            // ``ServerManager.activePort`` so a PortAllocator fallback
            // from the default port → +1 routes to the actual rapid-mlx
            // the child started. MUST run after the ensureServing await:
            // ``activePort`` is only assigned inside ``start``.
            if let server, server.activePort != 0 {
                client.baseURL = ChatStreamClient.loopbackURL(port: server.activePort)
            }

            if WorkAgent.gooseChat && gooseFallback == nil {
                do {
                    try await self.runGooseTurn(alias: alias, placeholderIndex: placeholderIndex, epoch: epoch,
                        globalInstruction: globalInstruction, conversationInstruction: chatInstruction)
                    return
                } catch {
                    self.activateGooseFallback(error)
                }
            }
            await self.runPlainChatTurn(
                alias: alias,
                initialPlaceholder: placeholderIndex,
                epoch: epoch,
                supportsImageInput: supportsImageInput,
                globalInstruction: globalInstruction,
                conversationInstruction: chatInstruction,
                memoryContext: memoryContext
            )
        }
    }

    private func runGooseTurn(alias: String, placeholderIndex: Int, epoch: Int,
                              globalInstruction: String, conversationInstruction: String) async throws {
        guard let server, let bearer = server.activeBearer, server.activePort > 0 else {
            throw WorkError.message("The local model is unavailable.")
        }
        try FileManager.default.createDirectory(at: gooseFolder, withIntermediateDirectories: true)
        let agent = WorkAgent.shared
        let priorIDs = Array(messages.prefix(placeholderIndex - 1)).map(\.id)
        let sessionBeforePrepare = agent.currentSessionID
        let canReuse = Self.canReuseGooseSession(
            activeConversationID: activeConversationID,
            sessionConversationID: gooseConversationID,
            currentContextIDs: gooseContextIDs,
            visibleContextIDs: priorIDs,
            sessionID: sessionBeforePrepare)
        agent.onChatEvent = { [weak self] event in
            guard let self, epoch == self.conversationEpoch else { return }
            self.applyGooseEvent(event, initialPlaceholder: placeholderIndex)
        }
        try await agent.prepare(folder: gooseFolder, rapidPort: server.activePort,
                                model: server.servingAlias ?? alias, bearer: bearer)
        guard epoch == conversationEpoch, !Task.isCancelled else {
            finishGooseCancellation(epoch: epoch)
            return
        }
        let sameHelperSession = agent.currentSessionID == sessionBeforePrepare
        let reuseSession = canReuse && sameHelperSession
        if !reuseSession {
            if sameHelperSession { try await agent.ensureNewGooseChatSession() }
            gooseConversationID = activeConversationID
        }
        let previous = Array(messages.prefix(placeholderIndex - 1))
        let replay = reuseSession ? "" : Self.gooseReplay(previous)
        let user = messages[placeholderIndex - 1]
        let instructions = [globalInstruction, conversationInstruction]
            .compactMap { Self.normalizedInstruction($0) }
        let instructionText = instructions.isEmpty ? "" : "Instructions for this chat:\n" + instructions.joined(separator: "\n") + "\n\n"
        let prompt = instructionText + replay + user.content + (try gooseAttachmentNote(for: user))
        await agent.prompt(prompt)
        guard epoch == conversationEpoch else { return }
        if Task.isCancelled {
            finishGooseCancellation(epoch: epoch)
            return
        }
        if let error = agent.error {
            Self.clearGooseFailedText(&messages[placeholderIndex])
            messages[placeholderIndex].status = .failed
            messages[placeholderIndex].errorMessage = Self.gooseFailureCopy
            messages[placeholderIndex].gooseErrorDetails = error
            lastError = Self.gooseFailureCopy
            lastFailureAlias = alias
            isStreaming = false
            inflight = nil
            gooseContextIDs = []
            return
        }
        for index in messages.indices where messages[index].role == .assistant && messages[index].status == .streaming {
            messages[index].status = .complete
        }
        isStreaming = false
        inflight = nil
        gooseContextIDs = messages.map(\.id)
    }

    static func gooseReplay(_ prior: [ChatMessage]) -> String {
        let turns = prior.filter {
            ($0.role == .user || $0.role == .assistant)
                && (!$0.content.isEmpty || $0.gooseParts?.contains(where: { $0.tool != nil }) == true)
        }.suffix(8).map { message in
            let tools = (message.gooseParts ?? []).compactMap(\.tool).suffix(6).map { card in
                "\(card.name): \(card.status)\(card.output.isEmpty ? "" : " — \(String(card.output.prefix(200)))")"
            }
            // Keep the conclusion of a grouped text/tool/text turn. Its
            // pre-tool narration can otherwise consume the whole replay cap.
            let text = message.role == .assistant
                ? String(message.content.suffix(1200)) : String(message.content.prefix(1200))
            let summary = tools.isEmpty ? text : text + "\nTools: " + tools.joined(separator: "; ")
            return "\(message.role == .user ? "User" : "Assistant"): \(summary)"
        }
        guard !turns.isEmpty else { return "" }
        return "Conversation so far (for context):\n" + turns.joined(separator: "\n") + "\n\nCurrent request: "
    }

    static func canReuseGooseSession(activeConversationID: UUID, sessionConversationID: UUID?,
                                     currentContextIDs: [UUID], visibleContextIDs: [UUID],
                                     sessionID: String?) -> Bool {
        sessionID != nil && sessionConversationID == activeConversationID
            && currentContextIDs == visibleContextIDs
    }

    private func finishGooseCancellation(epoch: Int) {
        guard epoch == conversationEpoch else { return }
        for index in messages.indices where messages[index].role == .assistant && messages[index].status == .streaming {
            var message = messages[index]
            Self.finaliseCancellation(message: &message)
            messages[index] = message
        }
        isStreaming = false
        inflight = nil
        gooseContextIDs = []
    }

    private func gooseAttachmentNote(for user: ChatMessage) throws -> String {
        guard !user.imageAttachments.isEmpty || !user.fileAttachments.isEmpty else { return "" }
        let directory = gooseFolder.appendingPathComponent("Attachments/\(user.id.uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var paths: [String] = []
        for image in user.imageAttachments {
            let name = URL(fileURLWithPath: image.filename).lastPathComponent
            let destination = directory.appendingPathComponent(name)
            try image.data.write(to: destination, options: .atomic)
            paths.append(destination.path)
        }
        for file in user.fileAttachments {
            let name = URL(fileURLWithPath: file.filename).deletingPathExtension().lastPathComponent + ".txt"
            let destination = directory.appendingPathComponent(name)
            try file.extractedText.write(to: destination, atomically: true, encoding: .utf8)
            paths.append(destination.path)
        }
        return "\nAttached files copied into this conversation folder:\n" + paths.joined(separator: "\n")
    }

    func applyGooseEvent(_ event: GooseChatEvent, initialPlaceholder: Int) {
        guard messages.indices.contains(initialPlaceholder) else { return }
        switch event {
        case .session(let id): gooseSessionID = id
        case .text(let chunk):
            guard !chunk.isEmpty else { break }
            if let details = messages[initialPlaceholder].gooseErrorDetails,
               details.contains("Ran into this error: Server error") {
                messages[initialPlaceholder].gooseErrorDetails = details + chunk
                break
            }
            if (messages[initialPlaceholder].content + chunk)
                .contains("Ran into this error: Server error") {
                messages[initialPlaceholder].gooseErrorDetails = messages[initialPlaceholder].content + chunk
                Self.clearGooseFailedText(&messages[initialPlaceholder])
                messages[initialPlaceholder].status = .failed
                messages[initialPlaceholder].errorMessage = Self.gooseFailureCopy
                lastError = Self.gooseFailureCopy
                break
            }
            messages[initialPlaceholder].content += chunk
            if messages[initialPlaceholder].gooseParts == nil { messages[initialPlaceholder].gooseParts = [] }
            if let last = messages[initialPlaceholder].gooseParts?.indices.last,
               messages[initialPlaceholder].gooseParts?[last].text != nil {
                messages[initialPlaceholder].gooseParts?[last].text? += chunk
            } else {
                messages[initialPlaceholder].gooseParts?.append(GooseTurnPart(text: chunk))
            }
        case .thinking(let chunk):
            messages[initialPlaceholder].reasoning += chunk
        case .tool(let id, let name, let input, let status, let output):
            if messages[initialPlaceholder].gooseParts == nil { messages[initialPlaceholder].gooseParts = [] }
            if let part = messages[initialPlaceholder].gooseParts?.firstIndex(where: { $0.tool?.id == id }),
               var card = messages[initialPlaceholder].gooseParts?[part].tool {
                if !name.isEmpty && name != "Tool" { card.name = name }
                if !input.isEmpty { card.input = input }
                if !output.isEmpty { card.output = String(output.prefix(12000)) }
                card.status = status
                messages[initialPlaceholder].gooseParts?[part].tool = card
            } else {
                messages[initialPlaceholder].gooseParts?.append(GooseTurnPart(
                    tool: GooseToolCard(id: id, name: name, input: input,
                                        output: String(output.prefix(12000)), status: status,
                                        folder: gooseFolder.path)))
            }
        case .permission(let request, let name, let state):
            goosePermissions[request.toolCallID] = (request, name)
            if messages[initialPlaceholder].gooseParts == nil { messages[initialPlaceholder].gooseParts = [] }
            if let part = messages[initialPlaceholder].gooseParts?.firstIndex(where: { $0.tool?.id == request.toolCallID }) {
                messages[initialPlaceholder].gooseParts?[part].tool?.permissionState = state
            } else {
                messages[initialPlaceholder].gooseParts?.append(GooseTurnPart(
                    tool: GooseToolCard(id: request.toolCallID, name: name,
                                        input: request.command, permissionState: state,
                                        folder: gooseFolder.path)))
            }
        case .error(let message):
            lastError = Self.gooseFailureCopy
            Self.clearGooseFailedText(&messages[initialPlaceholder])
            messages[initialPlaceholder].status = .failed
            messages[initialPlaceholder].errorMessage = Self.gooseFailureCopy
            messages[initialPlaceholder].gooseErrorDetails = message
        case .cancelled: break
        case .throughput(let rate):
            let message = messages[initialPlaceholder]
            messages[initialPlaceholder].stats = MessageStats(
                elapsedSeconds: max(0.1, Date().timeIntervalSince(message.createdAt)),
                charCount: message.content.count,
                promptTokens: nil,
                completionTokens: nil,
                timeToFirstTokenSeconds: nil,
                reasoningEmitted: !message.reasoning.isEmpty,
                engineTokensPerSecond: rate
            )
        }
    }

    static let gooseFailureCopy = String(localized: "Couldn't complete that request. Please try again.")

    private static func clearGooseFailedText(_ message: inout ChatMessage) {
        message.content = ""
        message.gooseParts = message.gooseParts?.filter { $0.tool != nil }
    }

    /// HF repo for the alias the next Send will start, supplied by the
    /// view from the catalog snapshot. Only used to install the
    /// bytes-on-disk progress monitor on a cold pull; a nil value just
    /// means a less informative progress strip, never a failure.
    var startupHFPath: String?

    /// Turn the streaming placeholder into a failure row when the
    /// model could not be brought up at all. Mirrors the shape
    /// ``runPlainChatTurn`` uses for a transport failure so the existing
    /// Retry affordance applies.
    ///
    /// Internal (not `private`) so ``StreamCancelTests`` can pin the
    /// banner-classification contract directly, as it does for
    /// ``finishStartupCancellation``.
    func finishWithStartupFailure(
        placeholderIndex: Int,
        alias: String,
        imageMessageID: UUID? = nil,
        epoch: Int? = nil
    ) {
        if let epoch, epoch != conversationEpoch { return }
        applyImageDeliveryEvent(
            messageID: imageMessageID,
            event: .abandoned,
            epoch: conversationEpoch
        )
        let genericMessage = "Couldn't start \(alias). Try again, or pick a different model in the box below."
        let message: String
        if let server,
           let startupFailure = server.startupFailure,
           case .crashed(let failedAlias, _) = server.state,
           failedAlias == alias {
            message = startupFailure.message
        } else {
            message = genericMessage
        }
        if var placeholder = currentMessage(index: placeholderIndex) {
            placeholder.status = .failed
            if placeholder.content.isEmpty { placeholder.content = message }
            updateMessage(at: placeholderIndex, with: placeholder)
        }
        lastError = message
        // Attribute the failure to the alias that actually failed, not to
        // whatever the picker happens to hold later.
        lastFailureAlias = alias
        // Classify the banner as a model-load failure (#590) rather than
        // letting it fall back to chatFailureKind("Couldn't start …") →
        // the generic `.requestFailed` diagnosis, whose Retry action just
        // re-runs the same failing start. `.modelLoadFailed` surfaces the
        // "check the model files or choose another model" copy + an Open
        // Model Management action, which matches the placeholder guidance.
        lastFailureKind = .modelLoadFailed
        isStreaming = false
    }

    /// The user hit Stop while the model was still being brought up — or
    /// in the instant it reached ``.ready``. Unlike
    /// ``finishWithStartupFailure`` this is a *deliberate* cancel, so it
    /// must NOT paint a "Couldn't start" failure row or raise the
    /// ``lastError`` banner: the same "Stop no longer masquerades as a
    /// failure" contract that governs mid-stream cancel applies to the
    /// cold-start path too.
    ///
    /// It resets the streaming state the ``runPlainChatTurn`` ``defer`` would
    /// normally own — without this reset ``isStreaming`` sticks ``true``
    /// and the ``guard !isStreaming`` at the top of ``send`` wedges every
    /// future send in every session — and finalises the (usually empty)
    /// placeholder through the shared ``finaliseCancellation`` contract so
    /// it resolves to ``.complete`` + "Stopped." instead of spinning as
    /// ``.streaming`` forever.
    ///
    /// Internal (not `private`) purely so ``StreamCancelTests`` can pin
    /// the state-transition contract directly, exactly as it pins
    /// ``finaliseCancellation``.
    func finishStartupCancellation(
        placeholderIndex: Int,
        imageMessageID: UUID? = nil,
        epoch: Int? = nil
    ) {
        if let epoch, epoch != conversationEpoch { return }
        applyImageDeliveryEvent(
            messageID: imageMessageID,
            event: .abandoned,
            epoch: conversationEpoch
        )
        if var placeholder = currentMessage(index: placeholderIndex) {
            Self.finaliseCancellation(message: &placeholder)
            updateMessage(at: placeholderIndex, with: placeholder)
        }
        isStreaming = false
    }

    /// Cancel the in-flight stream, finalising whatever the assistant
    /// has produced so far. The placeholder transitions to ``.complete``
    /// with whatever bytes already arrived.
    func stop() {
        cancelInflightWork()
    }

    /// Stop AND snapshot, synchronously, for the app-termination path.
    ///
    /// ``stop()`` only *requests* cancellation; the streaming task's own
    /// cleanup — ``isStreaming = false`` plus ``persistActive()`` — runs later
    /// on the main actor. During termination the main actor is then blocked
    /// for seconds reaping the server child, so that continuation never gets
    /// to run before ``ConversationStore.flush()``, and the partial final turn
    /// is written nowhere. Snapshot here instead, while we still hold the
    /// actor. The task's own cleanup remains harmless: it re-persists the same
    /// state if it ever gets to run.
    func stopAndPersist() {
        cancelInflightWork()
        guard isStreaming else { return }
        // App termination snapshots synchronously, before the cancelled
        // stream task can reach its catch block. No request remains live, so
        // every `.pending` image delivery is abandoned rather than persisted
        // as an attachment that can never be inherited again.
        for index in messages.indices where messages[index].imageDeliveryStatus == .pending {
            messages[index].imageDeliveryStatus = nil
        }
        // Finalise through the SHARED cancellation contract before snapshotting.
        // Persisting a message still marked ``.streaming`` writes a turn that
        // reopens after relaunch as a permanent typing indicator, with no live
        // task left to ever complete or cancel it. Same transition the normal
        // stop path uses: ``.complete`` + "Stopped.", keeping whatever bytes
        // already arrived.
        if let idx = messages.indices.last,
           messages[idx].role == .assistant,
           messages[idx].status == .streaming,
           var last = currentMessage(index: idx) {
            Self.finaliseCancellation(message: &last)
            updateMessage(at: idx, with: last)
        }
        isStreaming = false
        persistActive()
    }

    /// Drop a stale chat-level error banner once the server provably
    /// reaches ``.ready`` again. The banner's copy is advice about a
    /// PAST failure ("Couldn't reach the model. Restart it from the
    /// model bar…") — the moment a restart succeeds, that advice has
    /// been followed and the banner is a lie. 2026-07 dogfood: the
    /// banner survived a manual stop → start cycle and kept accusing
    /// a model that was back to Ready. ContentView calls this on the
    /// ``.ready`` transition.
    ///
    /// Guarded on ``isStreaming`` out of caution only: ``send()``
    /// already clears ``lastError`` at turn start, so a live stream
    /// should never coexist with a stale banner.
    func clearStaleErrorBanner() {
        guard !isStreaming else { return }
        lastError = nil
        lastFailureKind = nil
        lastFailureAlias = nil
    }

    /// Pure transformation applied to the assistant placeholder when
    /// the user clicks Stop mid-stream. Lifted out of
    /// ``runOneStream`` so the contract — partial content stays, the
    /// row is marked ``.complete`` (NOT ``.failed`` — failed gets the
    /// red Retry bubble and visually drops the partial reply) with a
    /// short "Stopped." footer — can be pinned by tests without
    /// standing up the full async stream loop.
    ///
    /// Codex audit r1 (ChatViewModel.swift:737): if the model emitted
    /// a ``finish_reason: tool_calls`` chunk JUST before the user hit
    /// Stop, ``message.toolCalls`` was populated by the stream-loop
    /// handler but the tool-loop will never execute them — clearing
    /// the field here prevents a stale ``assistant(tool_calls)`` row
    /// from going out on the next wire body with no matching
    /// ``role: "tool"`` results (which most chat templates 400 on).
    static func finaliseCancellation(message: inout ChatMessage) {
        message.status = .complete
        message.errorMessage = "Stopped."
        message.toolCalls = nil
    }

    /// True when ``error`` is a cancellation, whichever shape it
    /// arrives in. ``Task.cancel()`` surfaces as Swift's
    /// ``CancellationError`` when it interrupts one of our own
    /// suspension points — but when the task is parked inside
    /// URLSession's async machinery at that moment, URLSession
    /// reports it as ``URLError(.cancelled)`` (NSURLErrorDomain
    /// -999) instead. 2026-07 dogfood: Stop mid-stream took the
    /// generic failure path and painted "Couldn't reach the model.
    /// Restart it…" over a perfectly healthy model, because only
    /// the ``CancellationError`` shape was routed to
    /// ``finaliseCancellation``. Both shapes must land there.
    nonisolated static func isCancellation(_ error: Error) -> Bool {
        if error is CancellationError { return true }
        let ns = error as NSError
        return ns.domain == NSURLErrorDomain && ns.code == NSURLErrorCancelled
    }

    /// v0.4.35: filter "empty-prose assistant" turns out of the wire
    /// body. Lifted out of ``runPlainChatTurn`` so the contract — keep
    /// assistants whose prose is non-empty OR that carry tool_calls,
    /// drop the rest — can be pinned by tests. See the call site for
    /// the full motivation (model-switch silent-failure bug).
    nonisolated static func filterEmptyAssistantsForWire(_ messages: [ChatMessage]) -> [ChatMessage] {
        var filtered: [ChatMessage] = []
        for msg in messages {
            guard msg.role == .assistant else {
                filtered.append(msg)
                continue
            }
            let proseEmpty = msg.content
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .isEmpty
            let noToolCalls = (msg.toolCalls?.isEmpty ?? true)
            guard proseEmpty && noToolCalls else {
                filtered.append(msg)
                continue
            }

            // If Stop landed before the first token, the transcript contains
            // an unanswered user prompt followed by an empty assistant row
            // captioned "Stopped.". Dropping only the assistant leaves that
            // prompt as the last historical turn, so the next send answers
            // the cancelled request instead of the new one. Remove the pair
            // from the wire history while keeping both rows visible in the UI.
            if msg.errorMessage == "Stopped.", filtered.last?.role == .user {
                filtered.removeLast()
            }
        }
        return filtered
    }

    /// Issue #477: strip forward-incompatible ``.unknown``-role messages
    /// out of the outbound wire body. Such rows only arise from decoding a
    /// ``sessions.json`` written by a newer build (a role the OpenAI
    /// schema grew, or a value an older build can't map) — the load path
    /// degrades them to ``.unknown`` so the sidebar isn't wiped, and the
    /// UI renders them as a neutral system note. But serialising
    /// ``{"role":"unknown", ...}`` into the next request would make the
    /// server 400 the send, so they MUST NOT reach the wire. This is
    /// wire-only: the rows stay visible in the transcript.
    static func filterUnknownRolesForWire(_ messages: [ChatMessage]) -> [ChatMessage] {
        messages.filter { $0.role != .unknown }
    }

    /// Keeps the system row and latest user turn, then fills the
    /// remaining context newest-first.
    static func trimMessagesForContextWindow(
        _ messages: [ChatMessage],
        contextWindow: Int?,
        keepFraction: Double = 0.75
    ) -> [ChatMessage] {
        guard let ctx = contextWindow, ctx > 0 else { return messages }
        guard !messages.isEmpty else { return messages }
        let budget = max(1, Int(Double(ctx) * keepFraction))
        let perRowCost: (ChatMessage) -> Int = { msg in
            max(1, TokenEstimate.tokens(in: msg.modelContent))
        }
        let totalTokens = max(1, messages.reduce(0) { $0 + perRowCost($1) })
        if totalTokens <= budget { return messages }

        var system: ChatMessage? = nil
        var body = messages
        if body.first?.role == .system {
            system = body.removeFirst()
        }
        let systemTokens = system.map(perRowCost) ?? 0
        let bodyBudget = max(1, budget - systemTokens)

        guard let currentTurnStart = body.lastIndex(where: { $0.role == .user }) else {
            return system.map { [$0] } ?? []
        }
        var keep = Array(body[currentTurnStart...])
        var running = keep.reduce(0) { $0 + perRowCost($1) }
        for msg in body[..<currentTurnStart].reversed() {
            let cost = perRowCost(msg)
            if running + cost > bodyBudget { break }
            keep.insert(msg, at: 0)
            running += cost
        }

        while let first = keep.first, first.role != .user {
            keep.removeFirst()
        }
        if let sys = system {
            keep.insert(sys, at: 0)
        }
        return keep
    }

    /// v0.4.35: classify a terminal stream as a soft failure when it
    /// produced no visible text and no tool calls. Lifted out of
    /// ``runOneStream`` so the contract can be tested independently
    /// of the async URLProtocol stream — the call site reads exactly
    /// the same fields out of the streamed ``current`` snapshot.
    ///
    /// Returns ``nil`` when the terminal is a real completion (any
    /// prose or any tool call present). Returns a non-nil error
    /// message when zero-content + zero-tool-calls — the caller
    /// applies it as ``errorMessage`` and flips ``status`` to
    /// ``.failed``.
    static func zeroContentFailureMessage(
        proseContent: String,
        toolCalls: [ToolCall]?,
        finishReason: String?,
        thinkingEnabled: Bool = false
    ) -> String? {
        // Back-compat shim around the richer ``classifyTerminal``
        // helper. Callers that don't know about ``reasoningContent``
        // (or that only need the hard-failure subset) keep working
        // unchanged. Anything that needs the cycle-2 reasoning
        // fallback path uses ``classifyTerminal`` directly.
        let outcome = classifyTerminal(
            proseContent: proseContent,
            reasoningContent: "",
            toolCalls: toolCalls,
            finishReason: finishReason,
            thinkingEnabled: thinkingEnabled
        )
        switch outcome {
        case .realCompletion, .reasoningOnlyTruncated:
            return nil
        case .emptyTurnFailure(let copy):
            return copy
        }
    }

    /// Classification of one stream's terminal moment. Cycle-2 fix
    /// for the F-002 / verbose-reasoning UX gap (filed
    /// 2026-06-19) where a reasoning-only assistant turn that hits
    /// ``max_tokens`` mid-think landed as an empty failed bubble
    /// — even though ``reasoning_content`` was populated.
    ///
    /// Three buckets:
    ///   * ``.realCompletion`` — prose OR a tool call landed. No
    ///     action needed.
    ///   * ``.reasoningOnlyTruncated(hint:)`` — prose empty, NO tool
    ///     calls, ``finish_reason == "length"``, AND reasoning is
    ///     populated. Caller leaves the row ``.complete`` and the
    ///     reasoning lane visible (the UI auto-expands the
    ///     disclosure for this state). ``hint`` is an actionable
    ///     soft caption ("hit max_tokens mid-reasoning — raise the
    ///     budget") that renders in the secondary, NOT red, lane.
    ///   * ``.emptyTurnFailure(message:)`` — every other empty
    ///     terminal (stop / nil / length-without-reasoning). Caller
    ///     flips the row to ``.failed`` with the message as the red
    ///     caption, exactly as the pre-cycle-2 code did.
    enum TerminalOutcome: Equatable {
        case realCompletion
        case reasoningOnlyTruncated(hint: String)
        case emptyTurnFailure(message: String)
    }

    static func classifyTerminal(
        proseContent: String,
        reasoningContent: String,
        toolCalls: [ToolCall]?,
        finishReason: String?,
        thinkingEnabled: Bool = false
    ) -> TerminalOutcome {
        let proseEmpty = proseContent
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty
        let noToolCalls = (toolCalls?.isEmpty ?? true)
        if !(proseEmpty && noToolCalls) {
            return .realCompletion
        }
        let reasoningEmpty = reasoningContent
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty
        // Cycle-2 F-002 fallback: the model burned its budget inside
        // the reasoning trace but produced *something* — the user
        // should see the (truncated) trace, not a red empty bubble.
        // Only fires on ``finish_reason == "length"`` because a
        // clean ``"stop"`` with reasoning-only output is much more
        // likely a parser / chat-template bug than a budget hit,
        // and the existing "switching models" copy is the right
        // pointer in that case.
        if finishReason == "length" && !reasoningEmpty {
            return .reasoningOnlyTruncated(
                hint: "Hit the Max Tokens limit with the answer still inside the reasoning trace. Open the Reasoning section to see the partial trace, then raise Max Tokens (Settings → Sampling) or simplify the prompt to get a final answer."
            )
        }
        if finishReason == "length" {
            // #161: hybrid models with thinking ON routinely burn
            // the entire 4 K default ``max_tokens`` budget inside
            // ``<think>...</think>`` on a 4 B / 9 B-class model and
            // emit zero answer tokens. Point the user at the toggle
            // directly when that's the likely cause.
            if thinkingEnabled {
                return .emptyTurnFailure(
                    message: "Hit the Max Tokens limit with the answer still inside the reasoning trace. Turn off Show reasoning (Settings → Sampling), or raise Max Tokens."
                )
            }
            return .emptyTurnFailure(
                message: "Hit the Max Tokens limit before any output. Raise Max Tokens, or try a shorter prompt."
            )
        }
        return .emptyTurnFailure(
            message: "The model returned no text. This sometimes happens right after switching models — try Regenerate, or start a new session."
        )
    }

    /// Rewind the visible path to ``index`` — everything from there on is
    /// moved into ``branchedAway`` rather than discarded.
    ///
    /// This is the single primitive behind Regenerate, Retry, and editing a
    /// prompt. All three used to do `messages = Array(messages.prefix(idx))`,
    /// which destroyed the answer the user was replacing: a regeneration that
    /// came back worse than the original was unrecoverable. Setting the tail
    /// aside instead costs nothing at the call sites — the very next
    /// ``send`` appends under ``messages.last``, so the replacement turn
    /// lands as a SIBLING of the rewound one and both stay reachable through
    /// the switcher.
    private func rewindPath(to index: Int) {
        guard messages.indices.contains(index) else { return }
        branchedAway.append(contentsOf: messages[index...])
        messages = Array(messages.prefix(index))
        treeShapeVersion += 1
    }

    /// Edit a user turn: replace its prose and re-send from that point.
    ///
    /// The pre-edit prompt and every answer under it are kept as a sibling
    /// branch, so the original wording stays one `‹ 1/2 ›` click away.
    ///
    /// The replay stays on the CURRENT conversation id. An earlier build
    /// forked to a fresh id here so the pre-edit transcript survived as a
    /// recoverable branch, but the branch was indistinguishable from a real
    /// chat in the sidebar: every edit and every Retry silently spawned a
    /// duplicate row with the same title, so a few regenerations buried the
    /// history list under near-identical entries. Branching WITHIN the
    /// conversation is what that comment was reaching for — the alternatives
    /// now live inside the one row the user is looking at.
    @discardableResult
    func editUserMessage(
        id: UUID,
        newContent: String,
        alias: String,
        supportsImageInput: Bool? = nil
    ) -> Bool {
        guard !isStreaming else { return false }
        guard let idx = messages.firstIndex(where: { $0.id == id && $0.role == .user }) else { return false }
        let imageAttachments = messages[idx].imageAttachments
        let fileAttachments = messages[idx].fileAttachments
        let trimmed = newContent.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty || !imageAttachments.isEmpty || !fileAttachments.isEmpty else {
            return false
        }
        rewindPath(to: idx)
        send(
            trimmed,
            alias: alias,
            supportsImageInput: supportsImageInput,
            imageAttachments: imageAttachments,
            fileAttachments: fileAttachments
        )
        return true
    }

    /// Re-answer the user turn at ``userIndex``, keeping every answer that
    /// already hangs off it as an alternative.
    ///
    /// Rewinds to just AFTER the prompt — the prompt itself stays on the path
    /// and is reused — so the new answer is appended as its child and lands as
    /// a true sibling of the one being replaced. Rewinding past the prompt and
    /// re-sending its text instead would append a duplicate prompt, and the
    /// two answers would end up in different branches entirely.
    private func regenerateAnswer(
        afterUserAt userIndex: Int,
        alias: String,
        supportsImageInput: Bool? = nil
    ) {
        guard !isStreaming, messages.indices.contains(userIndex) else { return }
        let userMessage = messages[userIndex]
        guard !userMessage.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !userMessage.imageAttachments.isEmpty
                || !userMessage.fileAttachments.isEmpty else { return }

        // Regenerate and Retry are user-owned turns. Stop any optional work
        // about the answer they are replacing before the path is rewound.
        cancelInflightWork()
        rewindPath(to: userIndex + 1)
        beginAssistantTurn(
            alias: alias,
            supportsImageInput: supportsImageInput
                ?? ModelCatalogCache.supportsImageInput(forAlias: alias, binary: server?.binaryPath)
        )
    }

    /// Drop the most recent assistant turn and re-answer the user turn that
    /// preceded it. Powers the per-message Regenerate button under each
    /// assistant bubble. No-op while a stream is in flight.
    ///
    /// The replaced answer is kept as a sibling, not discarded — see
    /// ``rewindPath(to:)``.
    func regenerateLast(alias: String, supportsImageInput: Bool? = nil) {
        guard !isStreaming else { return }
        guard let lastUserIndex = messages.lastIndex(where: { $0.role == .user }) else { return }
        regenerateAnswer(
            afterUserAt: lastUserIndex,
            alias: alias,
            supportsImageInput: supportsImageInput
        )
    }

    /// Retry the turn that produced a specific assistant message. This is
    /// intentionally message-addressed: retrying an older response rewinds
    /// to the user prompt immediately before it instead of regenerating the
    /// latest turn by accident.
    @discardableResult
    func retryAssistantMessage(
        id: UUID,
        alias: String,
        supportsImageInput: Bool? = nil
    ) -> Bool {
        guard !isStreaming else { return false }
        guard let assistantIndex = messages.firstIndex(where: {
            $0.id == id && $0.role == .assistant
        }) else { return false }
        guard let userIndex = messages[..<assistantIndex].lastIndex(where: {
            $0.role == .user
        }) else { return false }

        let userMessage = messages[userIndex]
        guard !userMessage.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !userMessage.imageAttachments.isEmpty
                || !userMessage.fileAttachments.isEmpty else {
            return false
        }
        // In place, on the SAME conversation id — see ``editUserMessage``
        // for why the old fork-into-a-separate-chat behaviour was removed.
        // The retried answer and its replacement become siblings.
        regenerateAnswer(
            afterUserAt: userIndex,
            alias: alias,
            supportsImageInput: supportsImageInput
        )
        return true
    }

    // MARK: - Branch navigation

    /// The node whose sibling group represents ``id``'s alternatives.
    ///
    /// A regeneration branches at the turn that ANSWERS a prompt, but a tool
    /// round-trip writes several rows for one logical answer:
    /// `user → assistant(tool_calls) → tool → assistant(final)`. The branch
    /// therefore forks at the dispatch row, while the row the user reads —
    /// and the only one carrying a visible action bar — is the final answer.
    /// Asking for the final answer's own siblings finds none, so the switcher
    /// vanished exactly when a tool was used.
    ///
    /// Walking back to the first node after the owning user turn maps every
    /// row of a logical answer onto the same fork, so the control appears on
    /// the answer the user is looking at and both directions stay reachable.
    private func branchAnchor(for id: UUID, in tree: TreeSnapshot) -> ChatMessage? {
        guard let node = tree.byID[id] else { return nil }
        // A user turn branches on its own account (prompt edits).
        guard node.role != .user else { return node }
        var current = node
        // Same cycle guard as ``MessageTree/activePath``, and needed for the
        // same reason: that walk TRUNCATES a corrupt parent chain rather than
        // rejecting it, so a looping tree still reaches the transcript and
        // every rendered row asks for its branch position. Without the seen
        // set this walk spins on the main actor and the window hangs.
        var seen: Set<UUID> = [node.id]
        while let parentID = current.parentID, let parent = tree.byID[parentID] {
            if parent.role == .user { return current }
            guard seen.insert(parent.id).inserted else { return current }
            current = parent
        }
        return current
    }

    /// The alternatives available at ``id``, in stable order.
    ///
    /// A single-element result means the turn was never regenerated; the UI
    /// hides the switcher in that case rather than rendering a permanent
    /// `‹ 1/1 ›`.
    func siblings(of id: UUID) -> [ChatMessage] {
        let tree = treeSnapshot()
        guard let anchor = branchAnchor(for: id, in: tree) else { return [] }
        return tree.childGroup(of: anchor.parentID)
    }

    /// 1-based position of ``id`` among its alternatives, and how many there
    /// are — exactly what the `‹ 2/3 ›` control renders. ``nil`` when the
    /// turn has no alternatives.
    func branchPosition(of id: UUID) -> (index: Int, count: Int)? {
        let tree = treeSnapshot()
        guard let anchor = branchAnchor(for: id, in: tree) else { return nil }
        let group = tree.childGroup(of: anchor.parentID)
        guard group.count > 1, let offset = group.firstIndex(where: { $0.id == anchor.id }) else {
            return nil
        }
        return (offset + 1, group.count)
    }

    /// Switch the visible transcript to the alternative at ``offset`` within
    /// ``id``'s sibling group.
    ///
    /// Blocked mid-stream: the in-flight turn writes by index into
    /// ``messages``, so swapping the buffer under it would land tokens in the
    /// wrong branch. The UI disables the control for the same reason.
    @discardableResult
    func selectBranch(at offset: Int, forSiblingOf id: UUID) -> Bool {
        guard !isStreaming else { return false }
        let snapshot = treeSnapshot()
        // Resolve to the fork this row belongs to before indexing, so a tool
        // round-trip's final answer selects among the same alternatives its
        // switcher displays.
        guard let anchor = branchAnchor(for: id, in: snapshot) else { return false }
        let group = snapshot.childGroup(of: anchor.parentID)
        guard group.indices.contains(offset) else { return false }
        let target = group[offset]
        guard target.id != anchor.id else { return false }

        // Suggestions and a pending generated title describe the branch the
        // user is leaving. Cancel them before adopting another visible path,
        // otherwise switching back could resurrect an old rail.
        cancelInflightWork()
        // From here on the operation MUTATES: leave the snapshot (its node
        // values may be stale — see ``TreeSnapshot``) and work on the live
        // buffers, which always carry the finished content.
        let tree = liveTree()
        // Resolve DOWN from the chosen sibling: it is typically an interior
        // turn with its own continuation, and the transcript has to end at a
        // leaf. Prefer wherever the user last was inside that branch, falling
        // back to its newest tip for one never visited.
        let leaf = MessageTree.deepestLeaf(
            from: target.id,
            in: tree,
            preferring: branchChoices
        )
        // Record the step itself before adopting. ``adoptTree`` only sees the
        // path it resolves, and this edge — the parent choosing THIS sibling —
        // is the one the user just made.
        if let parent = target.parentID {
            branchChoices[parent] = target.id
        }
        adoptTree(tree, activeLeafID: leaf)
        // Persist WITHOUT touching updatedAt: switching branches is
        // navigation, not work, and should not reshuffle the sidebar — the
        // same reasoning as ``selectConversation``'s `touching: false`.
        persistActive(touching: false)
        return true
    }

    /// Move one alternative earlier / later in ``id``'s sibling group.
    /// Bounded, not wrapping: `‹` on the first branch is a no-op so the
    /// control's disabled state matches what it does.
    @discardableResult
    func stepBranch(from id: UUID, by delta: Int) -> Bool {
        let tree = treeSnapshot()
        guard let anchor = branchAnchor(for: id, in: tree) else { return false }
        let group = tree.childGroup(of: anchor.parentID)
        guard let current = group.firstIndex(where: { $0.id == anchor.id }) else { return false }
        let target = current + delta
        guard group.indices.contains(target) else { return false }
        return selectBranch(at: target, forSiblingOf: id)
    }

    /// How many turns ``deleteMessage(id:)`` would remove — the message plus
    /// everything beneath it, across every branch.
    ///
    /// The confirmation dialog needs this: deleting one visible bubble can
    /// take a dozen turns with it, and a user who has branched cannot see
    /// what is hanging off the other branches. Returns 0 for an unknown id.
    func deletionImpact(of id: UUID) -> Int {
        let tree = treeSnapshot()
        guard tree.byID[id] != nil else { return 0 }
        var collected: Set<UUID> = [id]
        var frontier: [UUID] = [id]
        while let current = frontier.popLast() {
            for child in tree.childGroup(of: current) where collected.insert(child.id).inserted {
                frontier.append(child.id)
            }
        }
        return collected.count
    }

    /// Confirmation title naming exactly what a delete would take.
    ///
    /// A bare "Delete message?" understates a deletion that silently removes
    /// a long follow-up thread and any branches hanging off it — and after a
    /// few regenerations some of those turns are not on screen for the user
    /// to count for themselves.
    ///
    /// Lives on the view model rather than the row that renders it because
    /// ``MessageRow`` is `private` to ``ChatView``; this is the part of the
    /// dialog worth pinning in a test.
    nonisolated static func deleteConfirmationTitle(turnCount: Int) -> String {
        turnCount <= 1
            ? "Delete this message?"
            : "Delete this message and the \(turnCount - 1) turn\(turnCount == 2 ? "" : "s") below it?"
    }

    /// Delete a turn and every alternative continuation beneath it. The
    /// producing prompt stays: a childless prompt is still something the user
    /// said, and the natural seed for a fresh regeneration.
    ///
    /// The whole subtree goes: re-parenting the orphans onto the deleted
    /// node's parent would splice together a prompt and an answer that never
    /// belonged to each other and present it as real history.
    @discardableResult
    func deleteMessage(id: UUID) -> Bool {
        guard !isStreaming else { return false }
        // Live values, not the snapshot: what survives the delete is re-split
        // and persisted, so it must carry the finished content of every row.
        let tree = liveTree()
        guard tree.contains(where: { $0.id == id }) else { return false }
        cancelInflightWork()
        let doomed = MessageTree.subtree(of: id, in: tree)
        let remaining = tree.filter { !doomed.contains($0.id) }
        guard !remaining.isEmpty else {
            // Deleting the root clears the conversation rather than leaving
            // an empty row behind.
            deleteConversation(activeConversationID)
            return true
        }
        // Land on the branch nearest the deletion — the parent's newest
        // surviving continuation — instead of jumping to an unrelated one.
        let parentID = tree.first(where: { $0.id == id })?.parentID
        // Drop navigation hints that pointed into the deleted subtree before
        // resolving, so a stale edge cannot steer the walk at the very fork
        // the user is standing on.
        branchChoices = branchChoices.filter { entry in
            !doomed.contains(entry.key) && !doomed.contains(entry.value)
        }
        let leaf = parentID.map {
            MessageTree.deepestLeaf(from: $0, in: remaining, preferring: branchChoices)
        } ?? MessageTree.defaultLeaf(in: remaining, preferring: branchChoices)
        adoptTree(MessageTree.promotingOrphans(remaining), activeLeafID: leaf)
        persistActive()
        // Preserve extracts still referenced by a surviving branch.
        let stillReferenced = Set(remaining.flatMap { $0.fileAttachments.map(\.id) })
        let orphaned = tree
            .filter { doomed.contains($0.id) }
            .flatMap { $0.fileAttachments.map(\.id) }
            .filter { !stillReferenced.contains($0) }
        documentCache.remove(contentsOf: Set(orphaned))
        return true
    }

    /// Same as ``regenerateLast(alias:)`` but brings up ``newAlias``
    /// first so the regenerated turn is answered by the newly selected
    /// model. Surfaces a friendly error and bails without dropping the
    /// transcript tail if the new model fails to load.
    func regenerateLast(usingAlias newAlias: String) async {
        guard !isStreaming else { return }
        let trimmed = newAlias.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        // The user's regeneration intent supersedes optional work even when
        // bringing up the replacement model takes time or ultimately fails.
        cancelInflightWork()
        if let server {
            let ok = await server.ensureServing(
                alias: trimmed,
                hfPath: nil,
                estimatedMemoryGB: nil,
                replacementGroup: .assistant
            )
            guard ok else {
                lastFailureKind = .modelLoadFailed
                lastError = FailureDiagnoser.diagnosis(
                    for: .modelLoadFailed,
                    modelAlias: trimmed
                ).message
                lastFailureAlias = trimmed
                return
            }
        }
        regenerateLast(
            alias: trimmed,
            supportsImageInput: server?.supportsImageInput(forAlias: trimmed) ?? false
        )
    }

    // MARK: - Plain chat stream

    /// Historical conversations can contain agent tool rows. Keep readable
    /// user and assistant prose while dropping obsolete tool envelopes before
    /// sending a new plain chat request.
    nonisolated static func plainChatHistory(_ history: [ChatMessage]) -> [ChatMessage] {
        let flattened = history.compactMap { original -> ChatMessage? in
            guard original.role != .tool, original.role != .unknown else { return nil }
            var message = original
            message.toolCalls = nil
            message.toolCallID = nil
            message.wireSuffix = nil
            return message
        }
        return filterEmptyAssistantsForWire(flattened)
    }

    private func runPlainChatTurn(
        alias: String,
        initialPlaceholder: Int,
        epoch: Int,
        supportsImageInput: Bool,
        globalInstruction: String = "",
        conversationInstruction: String = "",
        memoryContext: String? = nil
    ) async {
        let isTextOnlyTurn = messages[..<initialPlaceholder]
            .last(where: { $0.role == .user })?.imageAttachments.isEmpty ?? true
        defer {
            if epoch == conversationEpoch {
                isStreaming = false
                inflight = nil
                if isTextOnlyTurn, !Task.isCancelled,
                   let delivered = currentMessage(index: initialPlaceholder),
                   delivered.status == .complete,
                   delivered.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false {
                    onProductValueDelivered(.chatReply)
                }
            }
        }
        let requestInstant = Date()
        var history = Self.plainChatHistory(Array(messages.prefix(initialPlaceholder)))
        history = Self.addingInstructionLayers(
            to: history,
            dateContext: Self.currentDateContext(now: requestInstant),
            memoryContext: memoryContext,
            global: globalInstruction,
            conversation: conversationInstruction
        )
        let ctxWindow = ModelInfoCatalog.info(
            for: alias,
            hfRepo: nil,
            serverContextWindow: sampling?.activeContextWindow
        ).contextWindow
        history = Self.trimMessagesForContextWindow(history, contextWindow: ctxWindow)
        history = Self.stampingClockContext(on: history, answeringAt: requestInstant)
        let request: ChatStreamClient.Request
        if let sampling {
            let resolved = sampling.resolved(toolsEnabled: false)
            request = ChatStreamClient.Request(
                alias: alias,
                messages: history,
                temperature: resolved.temperature,
                topP: resolved.topP,
                maxTokens: resolved.maxTokens,
                repetitionPenalty: resolved.repetitionPenalty,
                repetitionPenaltyIsImplicitDefault: resolved.repetitionPenaltyIsImplicitDefault,
                enableThinking: resolved.enableThinking,
                supportsImageInput: supportsImageInput
            )
        } else {
            request = ChatStreamClient.Request(
                alias: alias,
                messages: history,
                enableThinking: false,
                supportsImageInput: supportsImageInput
            )
        }
        let outcome = await runOneStream(
            placeholderIndex: initialPlaceholder,
            request: request,
            epoch: epoch
        )
        if case .toolCallsPending = outcome,
           epoch == conversationEpoch,
           var message = currentMessage(index: initialPlaceholder) {
            message.toolCalls = nil
            message.status = .failed
            message.errorMessage = "The model returned a tool call that chat cannot use. Try again."
            updateMessage(at: initialPlaceholder, with: message)
        }
    }

    /// Merge current-date context, pre-existing, global, and conversation
    /// layers into one leading system row. Local chat templates often reject
    /// a second system message, so every caller must go through this
    /// transformation. Nothing request-scoped belongs here: the row is the
    /// head of every prompt, and any byte that changes between two turns
    /// costs the engine's prefix cache the whole conversation (that is why
    /// the wall clock rides each user row and the tool guidance rides the
    /// newest one — ``stampingClockContext``).
    nonisolated static func addingInstructionLayers(
        to messages: [ChatMessage],
        dateContext: String? = nil,
        memoryContext: String? = nil,
        global: String,
        conversation: String
    ) -> [ChatMessage] {
        var result = messages
        let existing = result.first?.role == .system ? result.removeFirst().content : nil
        var parts = [dateContext, existing]
            .compactMap { $0.flatMap(normalizedInstruction) }
        if let memoryContext, let memory = normalizedInstruction(memoryContext) {
            parts.append(memory)
        }
        if let global = normalizedInstruction(global) {
            parts.append("""
            [GLOBAL USER INSTRUCTIONS]
            These user preferences apply unless this conversation has a conflicting instruction:
            \(global)
            """)
        }
        if let conversation = normalizedInstruction(conversation) {
            parts.append("""
            [CONVERSATION INSTRUCTIONS - HIGHEST USER PRIORITY]
            These instructions apply only to this conversation. If they conflict with the global user instructions above, follow THESE conversation instructions. They do not override earlier application, safety, or tool instructions:
            \(conversation)
            """)
        }
        guard !parts.isEmpty else { return result }
        result.insert(
            ChatMessage(role: .system, content: parts.joined(separator: "\n\n"), status: .complete),
            at: 0
        )
        return result
    }

    /// Read-only preview of the Desktop-authored system prompt shared by
    /// Settings and the per-conversation editor. It deliberately uses the
    /// same assembly function as the wire path so displayed precedence and
    /// automatic context cannot drift from what Rapid sends.
    nonisolated static func effectiveSystemPrompt(
        dateContext: String? = nil,
        memoryContext: String? = nil,
        global: String,
        conversation: String
    ) -> String {
        addingInstructionLayers(
            to: [],
            // The preview shows what actually goes on the wire: the system
            // row carries the DATE only. The wall clock rides each user turn
            // as a wire-only trailer (``stampingClockContext``), so quoting it
            // here would show the reader a system row Rapid never sends. The
            // editor's caption tells them where the time went.
            dateContext: dateContext ?? currentDateContext(),
            memoryContext: memoryContext,
            global: global,
            conversation: conversation
        ).first?.content ?? ""
    }

    /// Request-time current-date context for the leading system row, so a small
    /// model does not guess "today" from training memory (issue #2330, where
    /// `qwen3.5-4b-4bit` answered "Friday, May 24, 2024" and then insisted it
    /// had no way to know the date). This supplies the Mac's authoritative
    /// local date as a system-prompt template variable at request time — the
    /// pattern peer desktop chat products use — rather than relying on the
    /// model to infer it must search for the date, and without adding a
    /// local-clock tool or touching tool routing.
    ///
    /// Injected per `send` (each request recomputes it against the live clock),
    /// so it cannot go stale across midnight, a time-zone change, a restored
    /// conversation, or a long-lived session. `now` and the calendar's time
    /// zone are injectable so tests can pin the exact output and cover
    /// rollover.
    ///
    /// The calendar day, time zone and nothing finer — the half of #2330's
    /// context that belongs in the leading system row.
    ///
    /// The wall CLOCK deliberately does not appear here; it rides each user
    /// turn instead (``clockContext(at:calendar:)`` +
    /// ``stampingClockContext(on:calendar:)``). The engine's prompt cache
    /// reuses a request only when its token prefix is byte-identical to a
    /// stored one, and the system row is the FIRST thing in every request: a
    /// minute-resolution string there rewrites the head of the prompt the
    /// moment the clock ticks, which turns an append-only follow-up into a
    /// full re-prefill of the whole conversation, document and all.
    ///
    /// Measured against this engine on an M3 Ultra (qwen3.8-27b-4bit, a
    /// 2.3k-token prompt): an append-only second turn with a byte-identical
    /// system row answers in 0.6 s; the same second turn with only the minute
    /// changed costs 7.3 s — the cold-start prefill, paid again. With an
    /// 8-page PDF attached the desktop was paying 15–17 s on EVERY follow-up.
    ///
    /// Day granularity still moves once per day, so a conversation spanning
    /// local midnight re-prefills exactly once. That is the price of keeping
    /// #2330's guarantee that the model never has to guess the date.
    nonisolated static func currentDateContext(
        now: Date = Date(),
        calendar inputCalendar: Calendar = .autoupdatingCurrent
    ) -> String {
        var gregorian = Calendar(identifier: .gregorian)
        gregorian.timeZone = inputCalendar.timeZone
        let zone = inputCalendar.timeZone
        let formatter = DateFormatter()
        formatter.calendar = gregorian
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = zone
        formatter.dateFormat = "EEEE, MMMM d, yyyy"
        let dateText = formatter.string(from: now)
        // Identifier only, no abbreviation. codex caught that "PST"/"PDT" is
        // instant-specific: it flips mid-day at a daylight-saving transition
        // and would re-prefill every open conversation at 2 a.m. twice a
        // year, in the one block this whole change exists to hold still. The
        // abbreviation still rides on each message trailer, where it is
        // describing a specific instant and is therefore correct there.
        return """
        [CURRENT DATE]
        Today is \(dateText) (\(zone.identifier)).
        """
    }

    /// One minute-resolution wall-clock stamp: "<full date> at <h:mm a>
    /// (<abbreviation>, <identifier>)". Shared by the "sent" trailer and the
    /// regenerate line so the two can never drift in format — and so
    /// comparing them for equality is a reliable "does this say anything new".
    nonisolated static func clockStampText(
        at instant: Date,
        calendar inputCalendar: Calendar = .autoupdatingCurrent
    ) -> String {
        var gregorian = Calendar(identifier: .gregorian)
        gregorian.timeZone = inputCalendar.timeZone
        let zone = inputCalendar.timeZone
        let formatter = DateFormatter()
        formatter.calendar = gregorian
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = zone
        formatter.dateFormat = "EEEE, MMMM d, yyyy 'at' h:mm a"
        let stampText = formatter.string(from: instant)
        let abbreviation = zone.abbreviation(for: instant) ?? zone.identifier
        return "\(stampText) (\(abbreviation), \(zone.identifier))"
    }

    /// The wall clock at `instant`, for the trailer stamped on a user turn.
    ///
    /// Split out of the system row for the prefix-cache reason documented on
    /// ``currentDateContext``. It keeps #2330's contract — the newest user row
    /// always carries the time it was sent, so the model is still told the
    /// current local time on every request and never answers a "what time is
    /// it" from training memory — while keeping the volatile tokens out of the
    /// prompt's head.
    nonisolated static func clockContext(
        at instant: Date,
        calendar inputCalendar: Calendar = .autoupdatingCurrent
    ) -> String {
        """
        [MESSAGE SENT]
        This message was sent \(clockStampText(at: instant, calendar: inputCalendar)).
        """
    }

    /// The extra trailer line a REGENERATE or RETRY adds to the turn it is
    /// re-answering, and only when it carries information: the question was
    /// asked at one instant and is being answered at a materially later one.
    ///
    /// Returns "" when both instants render to the same minute-resolution
    /// stamp, which is every ordinary send (the row was minted by this
    /// request) and every regenerate within the same minute. So the common
    /// path is byte-identical to having no such line at all.
    ///
    /// Why this exists: the reused row's "sent" stamp is honestly the original
    /// ask time, but a model asked "what time is it?" and re-answering it 45
    /// minutes later would otherwise report the stale clock — the exact
    /// regression codex caught, against the pre-change behaviour where the
    /// clock was recomputed per request. Appending rather than rewriting keeps
    /// the row's own stamp immutable.
    ///
    /// The cache cost is bounded and paid only by the user who regenerated:
    /// the line is gone again on the NEXT turn (which re-stamps from
    /// ``ChatMessage/createdAt``), so that turn's shared prefix ends just
    /// before this row instead of after it — a re-prefill of the last exchange,
    /// not of the conversation. The document in the first turn stays cached.
    nonisolated static func answeringNowLine(
        asked: Date,
        answeringAt: Date,
        calendar: Calendar = .autoupdatingCurrent
    ) -> String {
        let askedText = clockStampText(at: asked, calendar: calendar)
        let nowText = clockStampText(at: answeringAt, calendar: calendar)
        guard askedText != nowText else { return "" }
        return "\nThis answer is being generated \(nowText)."
    }

    /// Stamp every user row with the moment it was sent, as a wire-only
    /// trailer.
    ///
    /// The trailer reads "This message was sent <full date> at <time>", not
    /// "the current local time is …", and it carries the date as well as the
    /// clock. Both were codex findings on this PR, and both are about the
    /// same thing: every user row in the history wears one of these, so
    /// anything phrased as "current" would have the prompt asserting three
    /// contradictory current times, and a bare time on an old row would
    /// silently attach yesterday's clock to today's ``[CURRENT DATE]`` in a
    /// conversation that crossed midnight. A send-time stamp is true of every
    /// row forever, which is also exactly what makes it cacheable.
    ///
    /// The model still knows what time it is now: the newest user row is the
    /// one being sent, so its stamp IS the request time.
    ///
    /// Derived from each row's own ``ChatMessage/createdAt`` — which is
    /// persisted and immutable — rather than from "now", so the rendering of
    /// a turn never changes after the fact and each request is a strict
    /// EXTENSION of the one before it. That is the precise shape the engine's
    /// prompt cache requires: it reuses a stored entry only when that entry is
    /// a token-exact prefix of the new request.
    ///
    /// Stamping only the newest row was tried first and does not work. The
    /// trailer then has to be removed from that row on the following turn, the
    /// stored entry stops being a prefix, and reuse is lost exactly as it was
    /// with the clock in the system row. Measured on this engine
    /// (qwen3.8-27b-4bit, ~2.3k-token prompt, M3 Ultra): newest-row-only
    /// trailer 7.4 s on turn two; stamped-per-turn 0.8 s against 7.5 s cold.
    ///
    /// The cost is roughly twenty tokens per user turn, bought with the entire
    /// prefill of every follow-up — 15–17 s in the 0.14.1 document dogfood.
    ///
    /// What this means for the three ways a turn can be re-sent, because codex
    /// asked and the answer is not uniform:
    ///
    ///   * **A new send** mints a fresh row (``send(_:alias:…)`` builds a
    ///     ``ChatMessage`` with the default ``createdAt`` of `Date()`), so the
    ///     trailer is the live clock. This is the common case.
    ///   * **An edited send** goes through ``editUserMessage(id:newContent:…)``,
    ///     which rewinds the path and calls ``send`` — so it, too, mints a new
    ///     row and gets the live clock. Free: the edited text already broke
    ///     the shared prefix at that row, so there is no reuse to protect.
    ///   * **Regenerate / Retry** (``regenerateAnswer(afterUserAt:…)``) reuse
    ///     the existing user row deliberately — the question was asked then,
    ///     and re-answering it is not asking it again — so its "sent" stamp
    ///     keeps the original ask time. But re-answering "what time is it?"
    ///     three quarters of an hour later must not report the stale clock,
    ///     which is why ``answeringNowLine`` APPENDS the answer time to that
    ///     one row when the two instants differ. Append, not rewrite: the
    ///     row's own stamp stays immutable, the common regenerate (same
    ///     minute) adds nothing at all, and the bounded cache cost of the rare
    ///     late one is documented on that helper.
    nonisolated static func stampingClockContext(
        on messages: [ChatMessage],
        calendar: Calendar = .autoupdatingCurrent,
        answeringAt: Date? = nil
    ) -> [ChatMessage] {
        guard messages.contains(where: { $0.role == .user }) else { return messages }
        var result = messages
        let newestUserIndex = result.lastIndex { $0.role == .user }
        for index in result.indices where result[index].role == .user {
            var trailer = clockContext(
                at: result[index].createdAt,
                calendar: calendar
            )
            if index == newestUserIndex, let answeringAt {
                trailer += answeringNowLine(
                    asked: result[index].createdAt,
                    answeringAt: answeringAt,
                    calendar: calendar
                )
            }
            result[index].wireSuffix = trailer
        }
        return result
    }

    nonisolated static func normalizedInstruction(_ value: String) -> String? {
        CustomInstructionsConfig.normalized(value)
    }

    // MARK: - Single-stream driver

    /// Outcome of one streamed assistant turn.
    private enum StreamOutcome {
        /// Either we got a non-tool finish reason, the user pressed Stop,
        /// or the transport failed. Either way, no further automatic
        /// requests should fire — the chat loop is done.
        case terminal
        /// finish_reason: "tool_calls" with a non-empty call list.
        /// Caller runs the tools, appends results, opens a new
        /// placeholder, and re-enters the loop.
        case toolCallsPending([ToolCall])
    }

    private func runOneStream(
        placeholderIndex: Int,
        request: ChatStreamClient.Request,
        epoch: Int
    ) async -> StreamOutcome {
        var current = currentMessage(index: placeholderIndex)
            ?? ChatMessage(role: .assistant, status: .streaming)
        var capturedCalls: [ToolCall] = []
        var capturedFinish: String?
        // v0.4.12: stream-start timestamp. Captured here rather
        // than via ``current.createdAt`` because the placeholder
        // may have been inserted seconds ago by the tool-call
        // loop (between rounds), and the user reads "elapsed
        // time" as "time the model spent on THIS round."
        let streamStart = client.now()
        // #478: VoiceOver live-region feedback. Streaming replies are
        // otherwise silent to a screen-reader user — no start / progress
        // / completion signal. ``AssistantStreamAnnouncer`` is the pure,
        // throttled decision core; ``VoiceOverAnnouncer`` posts the
        // AppKit announcement. We read ``isVoiceOverEnabled`` ONCE here
        // and gate every announce path on it, so when VoiceOver is off
        // there is zero scanning / posting on the hot streaming path.
        let voiceOverActive = NSWorkspace.shared.isVoiceOverEnabled
        var announcer = AssistantStreamAnnouncer()
        // v0.4.13: capture server-reported usage if the server
        // honours ``stream_options.include_usage``. Either /
        // both fields may stay nil — non-conforming servers and
        // mid-stream cancellations leave them empty and the
        // stats caption falls back to the char-count estimate.
        var capturedPromptTokens: Int?
        var capturedCompletionTokens: Int?
        // The instant the first GENERATED token lands, on whichever channel
        // carries it. Everything before it is prefill (prompt processing),
        // which must not be charged to the decode rate the caption reports —
        // see ``MessageStats/decodeSeconds``.
        //
        // Reasoning counts. A thinking model emits its whole reasoning block
        // before the first prose token, and the server's `completion_tokens`
        // includes those tokens; starting the clock at the first `.content`
        // delta would divide every reasoning token by the prose-only window
        // and report a rate several times the real one.
        var firstTokenAt: ContinuousClock.Instant?
        do {
            // #17 desktop-half: thread the per-launch bearer through
            // every chat request. ``server.activeBearer`` rotates
            // each ServerManager.start() and clears on stop/crash,
            // so a stale leaked token is bounded to the live session.
            try await client.send(request, bearerToken: server?.activeBearer) { [weak self] event in
                guard let self else { return }
                switch event {
                case .firstToken(let at):
                    self.applyImageDeliveryEvent(
                        messageID: request.imageMessageID,
                        event: .accepted,
                        epoch: epoch
                    )
                    // The stream says the first generated token landed, on
                    // whichever lane carried it. Stamping per-lane here
                    // instead would miss a turn that opens with a tool-call
                    // fragment and time the later prose, reporting a decode
                    // window that excludes real generation.
                    if firstTokenAt == nil { firstTokenAt = at }
                case .content(let delta):
                    current.content += delta
                    // #478: announce the response start once, then the
                    // trailing un-announced sentence(s) on a throttled
                    // cadence. Both self-gate (start cue once; chunk only
                    // on a terminator + interval) so there is no
                    // per-token spam.
                    if voiceOverActive {
                        if let cue = announcer.firstTokenCue(fullContent: current.content) {
                            VoiceOverAnnouncer.announce(cue)
                        }
                        if let chunk = announcer.onDelta(
                            fullContent: current.content,
                            now: Date()
                        ) {
                            VoiceOverAnnouncer.announce(chunk)
                        }
                    }
                case .reasoning(let delta):
                    current.reasoning += delta
                case .toolCalls(let calls):
                    capturedCalls = calls
                    current.toolCalls = calls
                case .usage(let prompt, let completion):
                    capturedPromptTokens = prompt
                    capturedCompletionTokens = completion
                case .finished(let reason):
                    self.applyImageDeliveryEvent(
                        messageID: request.imageMessageID,
                        event: .accepted,
                        epoch: epoch
                    )
                    capturedFinish = reason
                    current.status = .complete
                    // v0.4.35 + cycle-2 (2026-06-19): classify the
                    // terminal moment via ``classifyTerminal``.
                    //
                    // Three buckets the helper distinguishes:
                    //
                    //   * ``.realCompletion`` — prose or tool call
                    //     landed. Leave the row ``.complete`` with no
                    //     caption.
                    //
                    //   * ``.emptyTurnFailure`` — every other empty
                    //     terminal. Flip to ``.failed`` with the red
                    //     caption, exact pre-cycle-2 behaviour.
                    //     Covers v0.4.35's three causes:
                    //       - ``reason == "length"`` — max_tokens hit
                    //         before the first token landed.
                    //       - ``reason == "stop"`` — clean stop with
                    //         empty content (model-switch history-tail
                    //         mismatch, or mid-warmup chat template).
                    //       - ``reason == nil`` — non-conforming
                    //         server emitted [DONE] with no choice
                    //         content.
                    //
                    //   * ``.reasoningOnlyTruncated`` — cycle-2
                    //     F-002 fix. ``reason == "length"`` AND
                    //     ``reasoning_content`` is populated AND the
                    //     visible ``content`` is empty. Don't flag as
                    //     a failure; surface a SOFT (secondary, NOT
                    //     red) caption pointing at Max Tokens and let
                    //     the ChatView's auto-expand path reveal the
                    //     partial reasoning trace. Without this branch
                    //     a reasoning model (phi-4-mini-reasoning,
                    //     deepseek_r1, nanbeige4.1) that exceeds the
                    //     budget mid-think left the user staring at
                    //     an empty red bubble even though useful
                    //     state was already on the row.
                    let outcome = ChatViewModel.classifyTerminal(
                        proseContent: current.content,
                        reasoningContent: current.reasoning,
                        toolCalls: current.toolCalls,
                        finishReason: reason,
                        thinkingEnabled: request.enableThinking
                    )
                    switch outcome {
                    case .realCompletion:
                        // Cycle-13 (2026-06-20) F-5: verbose-output
                        // length-truncation badge. A non-reasoning
                        // dense model (nemotron-30b-4bit and similar)
                        // that exhausts ``max_tokens`` mid-answer
                        // lands here as ``.realCompletion`` because
                        // ``content`` is non-empty — but the body the
                        // user sees is half-finished and reads as a
                        // real answer. Mark the row so the chat view
                        // can paint a subtle "Answer cut off (Max
                        // Tokens hit). Increase Max Tokens to see the
                        // rest." caption inline at the bottom of the
                        // bubble.
                        //
                        // The helper's gates make this a no-op for
                        // every other ``.realCompletion`` shape:
                        //   * clean ``stop`` / ``tool_calls`` — the
                        //     reason gate short-circuits.
                        //   * ``length`` + populated reasoning +
                        //     populated content — the reasoning-empty
                        //     gate short-circuits. This shape is
                        //     intentionally NOT badged: the row already
                        //     surfaces its reasoning disclosure and the
                        //     visible answer body landed, so the
                        //     "cut off mid-answer" framing of the
                        //     verbose-output badge doesn't apply.
                        //     ``reasoningTruncated`` does NOT cover this
                        //     shape either — its branch only fires on
                        //     empty content + populated reasoning + length
                        //     via ``.reasoningOnlyTruncated``.
                        //   * Codex r1 NIT (2026-06-20): comment used to
                        //     misattribute this case to PR #317; corrected
                        //     above.
                        current.contentTruncated = ChatMessage.shouldFlagContentTruncated(
                            content: current.content,
                            reasoning: current.reasoning,
                            finishReason: reason
                        )
                    case .reasoningOnlyTruncated(let hint):
                        current.errorMessage = hint
                        // Keep status .complete — the row carries
                        // useful reasoning state and the soft caption
                        // is informational, not an error. The
                        // .failed lane would paint the row red and
                        // hide the message behind Regenerate prompts.
                        // Codex r1 MAJOR-1: structural marker (not
                        // copy-string inference) so the chat-view
                        // can distinguish this case from
                        // ``finaliseCancellation``'s "Stopped." path,
                        // which ALSO lands as ``.complete`` with
                        // empty content + populated reasoning.
                        current.reasoningTruncated = true
                    case .emptyTurnFailure(let message):
                        current.errorMessage = message
                        current.status = .failed
                    }
                    // #478: announce the terminal moment ONCE, with a
                    // short cue — never the reply body. Skipped for the
                    // intermediate ``tool_calls`` finish (the turn is not
                    // over; a follow-up round streams next and announces
                    // its own completion). ``.failed`` speaks the error
                    // string; every other terminal is "Response complete".
                    if voiceOverActive, reason != "tool_calls" {
                        let terminal: AssistantStreamAnnouncer.Terminal =
                            current.status == .failed ? .failed : .complete
                        if let cue = announcer.onTerminal(
                            terminal,
                            errorMessage: current.errorMessage
                        ) {
                            VoiceOverAnnouncer.announce(cue)
                        }
                    }
                }
                self.writeStreamMessage(at: placeholderIndex, epoch: epoch, current)
            }
        } catch where Self.isCancellation(error) {
            applyImageDeliveryEvent(
                messageID: request.imageMessageID,
                event: .abandoned,
                epoch: epoch
            )
            // v0.4.29 pin: cancel MUST land as .complete (not .failed)
            // so the half-streamed content the user already saw stays
            // in the transcript with normal styling. Flipping to
            // .failed would paint a red "Retry" bubble and drop the
            // partial reply visually. Logic lifted into a static
            // helper so the contract is testable without async fan-out.
            ChatViewModel.finaliseCancellation(message: &current)
            writeStreamMessage(at: placeholderIndex, epoch: epoch, current)
            // #478: tell a screen-reader user the reply was stopped.
            if voiceOverActive, let cue = announcer.onTerminal(.cancelled, errorMessage: nil) {
                VoiceOverAnnouncer.announce(cue)
            }
            return .terminal
        } catch {
            let imageRejection = request.imageMessageID == nil
                ? nil
                : (error as? ChatStreamError)?.attachmentFailureMessage
            applyImageDeliveryEvent(
                messageID: request.imageMessageID,
                // A transient transport/busy/runtime failure gets one bounded
                // implicit retry. A typed rejection is terminal immediately.
                event: imageRejection == nil ? .transientFailure : .terminalRejection,
                epoch: epoch
            )
            current.status = .failed
            // Raw error → log for support; the user only ever sees
            // humanize()'s clean, jargon-free copy.
            print("[chat] stream failed: \(error.localizedDescription)")
            let failureKind = FailureDiagnoser.chatFailureKind(error: error)
            let actionable = imageRejection ?? FailureDiagnoser.diagnosis(
                for: failureKind,
                modelAlias: request.alias
            ).message
            current.errorMessage = actionable
            current.failureKind = failureKind
            lastFailureKind = failureKind
            lastError = actionable
            lastFailureAlias = request.alias
            writeStreamMessage(at: placeholderIndex, epoch: epoch, current)
            // #478: speak the failure to a screen-reader user (the error
            // string), so a silent red bubble isn't the only signal.
            if voiceOverActive, let cue = announcer.onTerminal(.failed, errorMessage: actionable) {
                VoiceOverAnnouncer.announce(cue)
            }
            return .terminal
        }
        // v0.4.12: end-of-stream stats. We attach for the plain-
        // text path (capturedFinish nil/"stop"/"length") AND for
        // the tool-call path — even though tool-call turns
        // continue to a follow-up round, the user sees the
        // elapsed time for THIS turn's reasoning + decision.
        // Skipped only when the message ended up ``.failed`` (a
        // crashed mid-stream "produced" 1.7 s and 41 chars but
        // that's noise, not throughput).
        if current.status == .complete && !current.content.isEmpty {
            let elapsed = streamStart.duration(to: client.now()).seconds
            current.stats = MessageStats(
                elapsedSeconds: elapsed,
                charCount: current.content.count,
                promptTokens: capturedPromptTokens,
                completionTokens: capturedCompletionTokens,
                timeToFirstTokenSeconds: firstTokenAt
                    .map { streamStart.duration(to: $0).seconds },
                reasoningEmitted: !current.reasoning.isEmpty
            )
            writeStreamMessage(at: placeholderIndex, epoch: epoch, current)
        }
        if capturedFinish == "tool_calls" && !capturedCalls.isEmpty {
            return .toolCallsPending(capturedCalls)
        }
        return .terminal
    }

    /// Map a raw transport / SSE error into an actionable single-
    /// sentence message the user can read without knowing what
    /// "URLError -1001" means. The categories match the realistic
    /// failure modes we hit against rapid-mlx + a local server:
    ///
    ///   * ``streamTruncated`` — server died mid-response (#896);
    ///     the dedicated copy already lives on the error itself.
    ///   * Timeout — model is generating but URLSession's idle
    ///     limit (600 s) expired; suggest raising max_tokens
    ///     budget or restarting.
    ///   * Connection refused / can't reach host — server isn't
    ///     listening; suggest Restart from the picker / banner.
    ///   * HTTP non-2xx — the request was rejected; show a plain,
    ///     actionable recovery message. The raw status code + server
    ///     body are engine internals and are NOT shown to the user
    ///     (they belong in the logs).
    ///   * Anything else — fall back to localizedDescription so a
    ///     rare error still has SOMETHING readable instead of an
    ///     empty bubble.
    nonisolated static func humanize(_ error: Error) -> String {
        if let chat = error as? ChatStreamError {
            switch chat {
            case .streamTruncated:
                // The stream ended early — almost always the model
                // crashing mid-reply. Give a plain recovery path; the
                // raw engine detail stays in the logs (principle: error
                // copy must be human + actionable).
                return "Rapid lost the model mid-reply — it may have crashed. Restart it from the model bar at the top and try again."
            case .httpStatus(_, let body):
                // #471: a genuine capacity rejection (out-of-memory
                // admission cap, or the server busy finishing another
                // reply) has a *specific* recovery the generic message
                // hides. The raw status code + body still never reach the
                // user — only the classification does; diagnostics stay in
                // the logs.
                switch capacityKind(from: body) {
                case .outOfMemory: return outOfMemoryMessage
                case .serverBusy: return serverBusyMessage
                case .none:
                    return "Rapid couldn't complete that request. Try again, or restart the model from the bar at the top."
                }
            case .transport(let message):
                // #471: a memory cap that trips mid-generation surfaces
                // here (the SSE stream had already opened), not as an HTTP
                // status. Same classification so the user gets the OOM /
                // busy recovery path regardless of *when* the cap fired.
                switch capacityKind(from: message) {
                case .outOfMemory: return outOfMemoryMessage
                case .serverBusy: return serverBusyMessage
                case .none:
                    return "Rapid lost its connection to the model. Restart it from the bar at the top and try again."
                }
            }
        }
        let ns = error as NSError
        if ns.domain == NSURLErrorDomain {
            switch ns.code {
            case NSURLErrorTimedOut:
                return "The model stopped responding (nothing for 10 minutes). It may be stuck — restart it from the model bar at the top, or try a shorter message."
            case NSURLErrorCannotConnectToHost, NSURLErrorCannotFindHost:
                return "Can't reach the model. Use the model bar at the top to restart it."
            case NSURLErrorNetworkConnectionLost:
                return "The model disconnected mid-reply. Restart it from the model bar at the top and try again."
            case NSURLErrorNotConnectedToInternet:
                return "macOS says the network is off, but Rapid runs entirely on your Mac, so this usually doesn't matter. Restart the model from the model bar at the top; if that fails, restart your Mac."
            default:
                // Don't surface the raw NSURLError code/body (e.g.
                // "NSURLErrorDomain error -1004") — the diagnostic is
                // logged at the call site; the user gets a clean path.
                return "Couldn't reach the model. Restart it from the model bar at the top and try again."
            }
        }
        // Anything else (system / library error — e.g. a decode failure,
        // a cancelled task) is NOT one of our authored user-facing errors;
        // its localizedDescription is a raw diagnostic. The caller already
        // logged the raw error, so the user gets a plain, actionable path.
        return "Rapid couldn't complete that request. Try again, or restart the model from the bar at the top."
    }

    /// #471: the capacity failure modes that deserve a *specific* recovery
    /// message instead of the generic "couldn't complete that request".
    enum CapacityKind: Equatable { case outOfMemory, serverBusy, none }

    /// Classify a raw sidecar error body/message into a capacity failure
    /// mode. Matching is substring + case-insensitive against the server's
    /// own wording — the ``BackpressureError`` text from the D-METAL-CAP
    /// memory-admission gate (a genuine OOM: the model + KV won't fit) and
    /// the max-concurrent-requests gate (transient busy). Engine internals
    /// never reach the user; only this classification does. The signals are
    /// distinctive phrases (not bare tokens like "oom") so they don't
    /// collide with unrelated errors.
    nonisolated static func capacityKind(from raw: String) -> CapacityKind {
        let t = raw.lowercased()
        // Genuine out-of-memory: the D-METAL-CAP admission gate rejected
        // the request because weights + projected KV exceed the GPU cap.
        // Signals are OOM-SPECIFIC on purpose — a bare "would exceed" would
        // collide with the context-length error ("... would exceed model
        // context ...") whose fix is a shorter *prompt*, not a smaller
        // model. The METAL-CAP 503 body still matches on the specific
        // phrases below.
        let memorySignals = [
            "metal-cap", "gpu_memory_utilization",
            "projected kv", "metal active", "out of memory",
            "insufficient memory",
        ]
        if memorySignals.contains(where: { t.contains($0) }) { return .outOfMemory }
        // Concurrency backpressure — the server is busy, not out of memory.
        // Waiting fixes it; a smaller model does not.
        let busySignals = ["max_concurrent_requests", "server is busy", "max concurrent"]
        if busySignals.contains(where: { t.contains($0) }) { return .serverBusy }
        return .none
    }

    /// #471 acceptance: a genuine OOM must read as human + actionable, not
    /// "Transport error / Internal error during streaming". Points at the
    /// two levers the user actually controls. (The #968 sidecar fix removes
    /// the *false* rejects; this copy is for the requests that truly don't
    /// fit.)
    nonisolated static let outOfMemoryMessage =
        "This reply needs more memory than your Mac has free right now. Try a smaller model from the bar at the top, or ask for a shorter response."

    /// #471: concurrency backpressure is transient — distinct from OOM so
    /// the user waits instead of needlessly downsizing their model.
    nonisolated static let serverBusyMessage =
        "Rapid is busy finishing another reply. Give it a moment, then try again."
}
