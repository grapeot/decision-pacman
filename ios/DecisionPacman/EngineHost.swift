import Foundation

/// Owns the on-device model. Loading and every decision run on one serial
/// queue, so requests that arrive while the model is still loading wait for it.
final class EngineHost: @unchecked Sendable {
    static let shared = EngineHost()

    private let queue = DispatchQueue(label: "decision-engine", qos: .userInitiated)
    private var engine: DecisionEngine?
    private var loadError: String?
    private(set) var loadSeconds: Double = 0

    static var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }
    static var modelURL: URL { documents.appendingPathComponent("model.gguf") }

    /// Starts loading in the background. Safe to call more than once.
    func loadIfNeeded() {
        queue.async { self.loadOnQueue() }
    }

    private func loadOnQueue() {
        if engine != nil { return }
        guard FileManager.default.fileExists(atPath: Self.modelURL.path) else {
            loadError = "No model on this device. Copy it into the app's Documents folder as model.gguf."
            return
        }
        let started = Date()
        do {
            engine = try DecisionEngine(modelPath: Self.modelURL.path)
            loadSeconds = Date().timeIntervalSince(started)
            loadError = nil
        } catch {
            loadError = error.localizedDescription
        }
    }

    /// Scores one rendered prompt. The reply is the dictionary the web page expects.
    func decide(prompt: String, options: Int, reply: @escaping @Sendable ([String: Any]) -> Void) {
        queue.async {
            self.loadOnQueue()
            guard let engine = self.engine else {
                reply(["error": self.loadError ?? "The model is not loaded."])
                return
            }
            do {
                let r = try engine.decide(prompt: prompt, optionCount: options)
                reply(["probabilities": r.probabilities, "tokens": r.tokens, "ms": r.milliseconds])
            } catch {
                reply(["error": error.localizedDescription])
            }
        }
    }

    /// Runs `body` with the engine on the engine queue.
    func withEngine<T: Sendable>(_ body: @escaping @Sendable (DecisionEngine) throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                self.loadOnQueue()
                guard let engine = self.engine else {
                    continuation.resume(throwing: NSError(domain: "DecisionPacman", code: 1,
                                                          userInfo: [NSLocalizedDescriptionKey: self.loadError ?? "The model is not loaded."]))
                    return
                }
                continuation.resume(with: Result { try body(engine) })
            }
        }
    }
}
