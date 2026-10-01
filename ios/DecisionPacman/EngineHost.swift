import Foundation

/// Owns the on-device model of the active player. Switching, loading, and every decision run on one
/// serial queue, so requests that arrive while a model is loading wait for it, and the previous model is
/// freed before the next one loads (phi4-mini alone needs about 3.3 GB).
final class EngineHost: @unchecked Sendable {
    static let shared = EngineHost()

    struct Status: Sendable {
        var player: Player
        var file: String?
        /// idle, loading, ready, missing, or error.
        var state: String
        var detail: String
        var loadSeconds: Double

        var dictionary: [String: Any] {
            var d: [String: Any] = ["player": player.rawValue, "state": state, "detail": detail,
                                    "loadSeconds": (loadSeconds * 100).rounded() / 100]
            if let file { d["file"] = file }
            return d
        }
    }

    private enum Engine {
        case decision(DecisionEngine)
        case chat(ChatMoveEngine)
    }

    private let queue = DispatchQueue(label: "decision-engine", qos: .userInitiated)
    private var active: Player = .finetuned
    private var engine: Engine?
    private var loadedKey: String?
    private var failedKey: String?

    private let lock = NSLock()
    private var currentStatus = Status(player: .finetuned, file: nil, state: "idle", detail: "", loadSeconds: 0)
    /// Called on the engine queue whenever the status changes.
    var onStatus: (@Sendable (Status) -> Void)?

    static var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }

    var status: Status {
        lock.lock()
        defer { lock.unlock() }
        return currentStatus
    }

    private func publish(_ status: Status) {
        lock.lock()
        currentStatus = status
        lock.unlock()
        onStatus?(status)
    }

    /// Makes `player` the active one: frees the current model, then loads the player's model in the background.
    func activate(_ player: Player) {
        queue.async {
            self.active = player
            self.failedKey = nil
            self.ensureLoaded()
        }
    }

    private func unload() {
        engine = nil  // deinit frees the context and the weights
        loadedKey = nil
    }

    /// Loads the active player's model unless it is already loaded. Returns an error message when it cannot.
    @discardableResult
    private func ensureLoaded() -> String? {
        let player = active
        let config = PlayerConfig.load()
        guard player.isOnDevice, let file = config.modelFile(for: player) else {
            unload()
            let hasKey = JevClient.hasKey
            publish(Status(player: player, file: nil, state: hasKey ? "ready" : "missing",
                           detail: hasKey ? "API key found" : "no API key: copy it to Documents/jev_key.txt", loadSeconds: 0))
            return nil
        }
        let context = config.chatContextLength
        let key = player == .finetuned ? "\(player.rawValue)/\(file)" : "\(player.rawValue)/\(file)/\(context)"
        if key == loadedKey, engine != nil { return nil }
        unload()
        let url = Self.documents.appendingPathComponent(file)
        guard FileManager.default.fileExists(atPath: url.path) else {
            let message = "No model: copy \(file) into the app's Documents folder."
            publish(Status(player: player, file: file, state: "missing", detail: message, loadSeconds: 0))
            return message
        }
        if key == failedKey { return status.detail }
        publish(Status(player: player, file: file, state: "loading", detail: "loading \(file)", loadSeconds: 0))
        let started = Date()
        do {
            switch player {
            case .finetuned: engine = .decision(try DecisionEngine(modelPath: url.path))
            case .phi4Mini, .qwen4b: engine = .chat(try ChatMoveEngine(modelPath: url.path, contextLength: context))
            case .jev: break
            }
            loadedKey = key
            let seconds = Date().timeIntervalSince(started)
            publish(Status(player: player, file: file, state: "ready", detail: String(format: "loaded in %.1f s", seconds), loadSeconds: seconds))
            return nil
        } catch {
            failedKey = key
            publish(Status(player: player, file: file, state: "error", detail: error.localizedDescription, loadSeconds: 0))
            return error.localizedDescription
        }
    }

    private func notActive(_ player: Player) -> [String: Any] {
        ["error": "\(player.title) is not the active player (\(active.title) is)."]
    }

    /// Fine-tuned decision model: scores one rendered prompt over `options` lettered options.
    func decide(player: Player, prompt: String, options: Int, reply: @escaping @Sendable ([String: Any]) -> Void) {
        queue.async {
            guard player == self.active else { return reply(self.notActive(player)) }
            if let message = self.ensureLoaded() { return reply(["error": message]) }
            guard case .decision(let engine) = self.engine else { return reply(["error": "\(player.title) has no decision model."]) }
            do {
                let r = try engine.decide(prompt: prompt, optionCount: options)
                reply(["probabilities": r.probabilities, "tokens": r.tokens, "ms": r.milliseconds])
            } catch {
                reply(["error": error.localizedDescription])
            }
        }
    }

    /// Plain chat model: generates {"move": ...} under `grammar` for one rendered chat prompt.
    func generate(player: Player, prompt: String, grammar: String, keys: [String], reply: @escaping @Sendable ([String: Any]) -> Void) {
        queue.async {
            guard player == self.active else { return reply(self.notActive(player)) }
            if let message = self.ensureLoaded() { return reply(["error": message]) }
            guard case .chat(let engine) = self.engine else { return reply(["error": "\(player.title) has no chat model."]) }
            do {
                let r = try engine.generate(prompt: prompt, grammar: grammar, keys: keys)
                var out: [String: Any] = ["text": r.text, "tokens": r.promptTokens, "outputTokens": r.outputTokens,
                                          "prefillMs": r.prefillMilliseconds, "ms": r.milliseconds]
                if let p = r.probabilities { out["probabilities"] = p }
                reply(out)
            } catch {
                reply(["error": error.localizedDescription])
            }
        }
    }
}
