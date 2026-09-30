// Runs the iOS app's inference and benchmark code on macOS against the same prompts.
//   see ios/scripts/mac_check.sh
import Foundation
import llama

func runCheck() throws {
    let args = CommandLine.arguments
    let t0 = Date()
    let engine = try DecisionEngine(modelPath: args[1])
    let loadSeconds = Date().timeIntervalSince(t0)
    let prompts = try JSONDecoder().decode([BenchPrompt].self, from: Data(contentsOf: URL(fileURLWithPath: args[2])))
    let report = try Benchmark.run(engine: engine, prompts: prompts, rounds: 3, device: "mac", model: args[1], loadSeconds: loadSeconds)
    print(String(format: "load %.2fs  p50 %.1f ms  p90 %.1f ms  mean %.1f ms  same choice %d/%d  max |dp| %.4f",
                 report.loadSeconds, report.p50, report.p90, report.mean, report.sameChoice, report.total, report.maxProbabilityDiff))
}

// Keep the engine inside a function so it is freed before the process exits;
// Metal resources released during exit-time teardown trip a ggml assertion.
try runCheck()
llama_backend_free()
