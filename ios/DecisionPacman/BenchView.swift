import SwiftUI

@MainActor
final class BenchModel: ObservableObject {
    @Published var status = ""
    @Published var busy = false
    @Published var report: BenchReport?
    @Published var progressLine = ""

    init() {
        refresh()
    }

    func refresh() {
        let attrs = try? FileManager.default.attributesOfItem(atPath: EngineHost.modelURL.path)
        if let size = attrs?[.size] as? Int64 {
            status = "Model on device (\(ByteCountFormatter.string(fromByteCount: size, countStyle: .file)))"
        } else {
            status = "No model on this device. Copy it into the app's Documents folder as model.gguf."
        }
    }

    @discardableResult
    func run(rounds: Int = 3, runID: String? = nil) async -> BenchReport? {
        busy = true
        status = "Benchmarking…"
        defer { busy = false }
        do {
            let prompts = try Benchmark.loadPrompts()
            let device = Self.deviceIdentifier()
            let report = try await EngineHost.shared.withEngine { engine in
                try Benchmark.run(engine: engine, prompts: prompts, rounds: rounds, device: device, model: "model.gguf",
                                  loadSeconds: EngineHost.shared.loadSeconds) { line in
                    Task { @MainActor in self.progressLine = line }
                }
            }
            self.report = report
            status = String(format: "p50 %.0f ms, p90 %.0f ms", report.p50, report.p90)
            let name = "bench_\(runID ?? ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")).json"
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: EngineHost.documents.appendingPathComponent(name))
            return report
        } catch {
            status = "Benchmark failed: \(error.localizedDescription)"
            return nil
        }
    }

    /// decisionpacman://bench?run_id=<id> runs the benchmark and writes Documents/bench_<id>.json.
    func handle(url: URL) {
        guard url.scheme == "decisionpacman", url.host == "bench" else { return }
        let runID = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.first { $0.name == "run_id" }?.value
        let safeID = runID?.filter { $0.isLetter || $0.isNumber || $0 == "-" }
        Task { await run(runID: safeID) }
    }

    nonisolated static func deviceIdentifier() -> String {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { raw in String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self) }
    }
}

struct BenchView: View {
    @EnvironmentObject var bench: BenchModel

    var body: some View {
        NavigationStack {
            Form {
                Section("Model") {
                    Text(bench.status).font(.footnote).foregroundStyle(.secondary)
                    Button("Run benchmark (20 decisions × 3)") { Task { await bench.run() } }
                        .disabled(bench.busy)
                    if bench.busy && !bench.progressLine.isEmpty {
                        Text(bench.progressLine).font(.footnote.monospaced())
                    }
                }
                if let r = bench.report {
                    Section("Result") {
                        row("Device", r.device)
                        row("Model load", String(format: "%.2f s", r.loadSeconds))
                        row("Latency p50", String(format: "%.0f ms", r.p50))
                        row("Latency p90", String(format: "%.0f ms", r.p90))
                        row("Mean", String(format: "%.0f ms", r.mean))
                        row("Same choice as reference", "\(r.sameChoice)/\(r.total)")
                        row("Max probability difference", String(format: "%.4f", r.maxProbabilityDiff))
                    }
                }
            }
            .navigationTitle("Latency")
        }
        .onAppear { bench.refresh() }
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value).monospacedDigit().foregroundStyle(.secondary)
        }
    }
}
