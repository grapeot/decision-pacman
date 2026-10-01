import AVFoundation
import SwiftUI

/// The active player, shared by the picker, the control URL route, and the page.
@MainActor
final class PlayerStore: ObservableObject {
    static let shared = PlayerStore()
    private static let defaultsKey = "player"

    @Published private(set) var active: Player
    @Published private(set) var engine: EngineHost.Status

    private init() {
        let saved = UserDefaults.standard.string(forKey: Self.defaultsKey).flatMap(Player.init(rawValue:))
        active = saved ?? .finetuned
        engine = EngineHost.shared.status
        EngineHost.shared.onStatus = { status in
            Task { @MainActor in PlayerStore.shared.engine = status }
        }
        // Start loading right away; decisions wait for it.
        EngineHost.shared.activate(active)
    }

    /// Switches players: the engine frees the previous on-device model before loading the next one.
    func select(_ player: Player) {
        guard player != active else { return }
        active = player
        UserDefaults.standard.set(player.rawValue, forKey: Self.defaultsKey)
        EngineHost.shared.activate(player)
        WebViewHolder.shared.setPlayer(player)
    }

    /// One line under the picker: the model file and whether it is loaded.
    var statusLine: String {
        guard engine.player == active else { return active.title }
        let file = engine.file.map { " · \($0)" } ?? ""
        return "\(active.title)\(file) · \(engine.state == "ready" ? engine.detail : "\(engine.state): \(engine.detail)")"
    }
}

struct ContentView: View {
    @ObservedObject private var store = PlayerStore.shared

    var body: some View {
        VStack(spacing: 4) {
            Picker("Player", selection: Binding(get: { store.active }, set: { store.select($0) })) {
                ForEach(Player.allCases) { Text($0.shortLabel).tag($0) }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 12)
            Text(store.statusLine)
                .font(.caption2.monospaced())
                .foregroundStyle(store.engine.state == "ready" || store.engine.player != store.active ? Color.secondary : Color.orange)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
                .padding(.horizontal, 12)
            GameView(initialPlayer: store.active)
        }
        .background(Color.black)
    }
}

@main
struct DecisionPacmanApp: App {
    init() {
        // Game sound follows the ring/silent switch and mixes with other audio.
        try? AVAudioSession.sharedInstance().setCategory(.ambient)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.dark)
                .onOpenURL { url in WebViewHolder.shared.handle(url: url) }
        }
    }
}
