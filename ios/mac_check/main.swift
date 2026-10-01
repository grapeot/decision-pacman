// Runs the iOS app's inference engines on macOS against fixed game prompts.
//   mac_check decision <model.gguf> bench_prompts.json   fine-tuned decision model vs Ollama's recorded probabilities
//   mac_check chat <model.gguf> chat_prompts.json        plain chat model under the move grammar vs Ollama's recorded answers
// See ios/scripts/mac_check.sh.
import Foundation
import llama

struct BenchPrompt: Decodable {
    let prompt: String
    let keys: [String]
    let expected: [Double]
}

struct ChatPrompt: Decodable {
    struct Reference: Decodable {
        let choice: String
        let probabilities: [Double]?
        let inputTokens: Int?
    }
    let prompt: String
    let grammar: String
    let keys: [String]
    let reference: Reference
}

func percentile(_ xs: [Double], _ p: Double) -> Double {
    let s = xs.sorted()
    return s[min(s.count - 1, Int(Double(s.count) * p))]
}

func argmax(_ xs: [Double]) -> Int { xs.indices.max { xs[$0] < xs[$1] }! }

func load<T: Decodable>(_ type: T.Type, _ path: String) throws -> T {
    try JSONDecoder().decode(type, from: Data(contentsOf: URL(fileURLWithPath: path)))
}

/// Same check as the app's old Latency tab: every prompt `rounds` times after a warm-up.
func checkDecision(model: String, prompts path: String, rounds: Int = 3) throws {
    let t0 = Date()
    let engine = try DecisionEngine(modelPath: model)
    let loadSeconds = Date().timeIntervalSince(t0)
    let prompts = try load([BenchPrompt].self, path)
    for p in prompts.prefix(2) { _ = try engine.decide(prompt: p.prompt, optionCount: p.keys.count) }
    var times: [Double] = []
    var same = 0
    var maxDiff = 0.0
    for _ in 0..<rounds {
        for p in prompts {
            let r = try engine.decide(prompt: p.prompt, optionCount: p.keys.count)
            times.append(r.milliseconds)
            if argmax(r.probabilities) == argmax(p.expected) { same += 1 }
            maxDiff = max(maxDiff, zip(r.probabilities, p.expected).map { abs($0 - $1) }.max() ?? 0)
        }
    }
    print(String(format: "decision: load %.2fs  p50 %.1f ms  p90 %.1f ms  same choice %d/%d  max |dp| %.4f",
                 loadSeconds, percentile(times, 0.5), percentile(times, 0.9), same, times.count, maxDiff))
}

func checkChat(model: String, prompts path: String) throws {
    let t0 = Date()
    // MAC_CHECK_CONTEXT=2048 runs phi4-mini with LongRoPE's short factors instead of the long ones Ollama used.
    let context = ProcessInfo.processInfo.environment["MAC_CHECK_CONTEXT"].flatMap { UInt32($0) } ?? ChatMoveEngine.defaultContextLength
    let engine = try ChatMoveEngine(modelPath: model, contextLength: context)
    let loadSeconds = Date().timeIntervalSince(t0)
    let prompts = try load([ChatPrompt].self, path)
    // Warm-up. The first answer also builds the option-token tables (one pass over the vocabulary per option).
    let first = try engine.generate(prompt: prompts[0].prompt, grammar: prompts[0].grammar, keys: prompts[0].keys)
    var times: [Double] = [], prefill: [Double] = [], perStep: [Double] = [], outTokens: [Double] = []
    var legal = 0, same = 0, tokenMatch = 0, withProbabilities = 0
    var diffs: [Double] = []
    for (i, p) in prompts.enumerated() {
        let r = try engine.generate(prompt: p.prompt, grammar: p.grammar, keys: p.keys)
        // The same legality check as the page: the answer parses as JSON and its move is an option.
        let parsed = (try? JSONSerialization.jsonObject(with: Data(r.text.utf8))) as? [String: Any]
        let isLegal = (parsed?["move"] as? String).map { p.keys.contains($0) } ?? false
        if isLegal { legal += 1 }
        if r.move == p.reference.choice { same += 1 }
        if r.promptTokens == p.reference.inputTokens { tokenMatch += 1 }
        if let mine = r.probabilities, let theirs = p.reference.probabilities {
            withProbabilities += 1
            diffs.append(zip(mine, theirs).map { abs($0 - $1) }.max() ?? 0)
        }
        times.append(r.milliseconds)
        prefill.append(r.prefillMilliseconds)
        outTokens.append(Double(r.outputTokens))
        if r.outputTokens > 1 { perStep.append((r.milliseconds - r.prefillMilliseconds) / Double(r.outputTokens - 1)) }
        let probs = r.probabilities.map { $0.map { String(format: "%.2f", $0) }.joined(separator: " ") } ?? "-"
        print(String(format: "%2d %@ -> %@ (ollama %@)  [%@]  %d+%d tokens  %.0f ms", i, p.keys.joined(separator: "/"), r.text,
                     p.reference.choice, probs, r.promptTokens, r.outputTokens, r.milliseconds))
    }
    print(String(format: "chat: context %d  first answer %.0f ms", context, first.milliseconds))
    print(String(format: "chat: load %.2fs  legal %d/%d  same move as Ollama %d/%d  prompt tokens equal to Ollama's %d/%d",
                 loadSeconds, legal, prompts.count, same, prompts.count, tokenMatch, prompts.count))
    print(String(format: "chat: latency p50 %.0f ms  p90 %.0f ms  prefill p50 %.0f ms  output tokens p50 %.0f  per decode step p50 %.1f ms",
                 percentile(times, 0.5), percentile(times, 0.9), percentile(prefill, 0.5), percentile(outTokens, 0.5),
                 perStep.isEmpty ? 0 : percentile(perStep, 0.5)))
    if !diffs.isEmpty {
        print(String(format: "chat: probabilities on %d prompts, max |dp| p50 %.4f  max %.4f", withProbabilities,
                     percentile(diffs, 0.5), diffs.max()!))
    }
}

func run() throws {
    let args = CommandLine.arguments
    guard args.count == 4 else {
        print("usage: mac_check decision|chat <model.gguf> <prompts.json>")
        exit(2)
    }
    switch args[1] {
    case "decision": try checkDecision(model: args[2], prompts: args[3])
    case "chat": try checkChat(model: args[2], prompts: args[3])
    default: print("unknown mode \(args[1])"); exit(2)
    }
}

// Keep the engines inside functions so they are freed before the process exits;
// Metal resources released during exit-time teardown trip a ggml assertion.
try run()
llama_backend_free()
