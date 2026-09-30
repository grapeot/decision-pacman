import AVFoundation
import SwiftUI

@main
struct DecisionPacmanApp: App {
    @StateObject private var bench = BenchModel()

    init() {
        // Start loading the model right away; decisions wait for it.
        EngineHost.shared.loadIfNeeded()
        // Game sound follows the ring/silent switch and mixes with other audio.
        try? AVAudioSession.sharedInstance().setCategory(.ambient)
    }

    var body: some Scene {
        WindowGroup {
            TabView {
                GameView()
                    .background(Color.black)
                    .tabItem { Label("Play", systemImage: "gamecontroller") }
                BenchView()
                    .tabItem { Label("Latency", systemImage: "stopwatch") }
            }
            .environmentObject(bench)
            .preferredColorScheme(.dark)
            .onOpenURL { url in
                bench.handle(url: url)
                WebViewHolder.shared.handle(url: url)
            }
        }
    }
}
