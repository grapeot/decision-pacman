import Foundation

struct BenchPrompt: Decodable, Sendable {
    let prompt: String
    let keys: [String]
    let letters: [String]
    let tokens: Int
    let expected: [Double]
}

struct BenchRow: Codable, Sendable {
    let index: Int
    let milliseconds: Double
    let tokens: Int
    let choice: String
    let expectedChoice: String
    let maxProbabilityDiff: Double
}

struct BenchReport: Codable, Sendable {
    let device: String
    let model: String
    let rounds: Int
    let loadSeconds: Double
    let p50: Double
    let p90: Double
    let mean: Double
    let sameChoice: Int
    let total: Int
    let maxProbabilityDiff: Double
    let rows: [BenchRow]
}

enum Benchmark {
    static func loadPrompts() throws -> [BenchPrompt] {
        guard let url = Bundle.main.url(forResource: "bench_prompts", withExtension: "json") else {
            throw CocoaError(.fileNoSuchFile)
        }
        return try JSONDecoder().decode([BenchPrompt].self, from: Data(contentsOf: url))
    }

    /// Runs every prompt `rounds` times after one warm-up pass and summarizes latency and agreement.
    static func run(engine: DecisionEngine, prompts: [BenchPrompt], rounds: Int, device: String, model: String, loadSeconds: Double,
                    progress: @escaping @Sendable (String) -> Void = { _ in }) throws -> BenchReport {
        for p in prompts.prefix(2) { _ = try engine.decide(prompt: p.prompt, optionCount: p.keys.count) }
        var rows: [BenchRow] = []
        for round in 0..<rounds {
            for (i, p) in prompts.enumerated() {
                let r = try engine.decide(prompt: p.prompt, optionCount: p.keys.count)
                let choice = p.keys[r.probabilities.indices.max { r.probabilities[$0] < r.probabilities[$1] }!]
                let expectedChoice = p.keys[p.expected.indices.max { p.expected[$0] < p.expected[$1] }!]
                let diff = zip(r.probabilities, p.expected).map { abs($0 - $1) }.max() ?? 0
                rows.append(BenchRow(index: i, milliseconds: r.milliseconds, tokens: r.tokens, choice: choice, expectedChoice: expectedChoice, maxProbabilityDiff: diff))
                progress("round \(round + 1)/\(rounds), prompt \(i + 1)/\(prompts.count): \(Int(r.milliseconds)) ms")
            }
        }
        let times = rows.map(\.milliseconds).sorted()
        return BenchReport(
            device: device, model: model, rounds: rounds, loadSeconds: loadSeconds,
            p50: times[times.count / 2], p90: times[Int(Double(times.count) * 0.9)],
            mean: times.reduce(0, +) / Double(times.count),
            sameChoice: rows.filter { $0.choice == $0.expectedChoice }.count, total: rows.count,
            maxProbabilityDiff: rows.map(\.maxProbabilityDiff).max() ?? 0, rows: rows)
    }
}
