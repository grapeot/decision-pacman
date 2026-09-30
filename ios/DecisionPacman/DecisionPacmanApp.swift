import SwiftUI

@main
struct DecisionPacmanApp: App {
    @StateObject private var bench = BenchModel()

    init() {
        // Start loading the model right away; decisions wait for it.
        EngineHost.shared.loadIfNeeded()
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
            .onOpenURL { bench.handle(url: $0) }
        }
    }
}
