import Foundation

/// The app's players. The raw values are shared with src/agent/players.ts, the page's `?model=`
/// parameter, and `decisionpacman://control?model=`.
enum Player: String, CaseIterable, Identifiable, Sendable {
    case finetuned
    case phi4Mini = "phi4-mini"
    case jev

    var id: String { rawValue }

    /// Segment title in the picker.
    var shortLabel: String {
        switch self {
        case .finetuned: return "0.8B tuned"
        case .phi4Mini: return "phi4-mini"
        case .jev: return "Jev"
        }
    }

    var title: String {
        switch self {
        case .finetuned: return "Distilled 0.8B (Qwen 27B teacher), on device"
        case .phi4Mini: return "phi4-mini 3.8B, on device"
        case .jev: return "Jev, cloud"
        }
    }

    var isOnDevice: Bool { self != .jev }
}

/// Which model file each on-device player loads from the app's Documents. The fine-tuned player uses
/// pacman-0.8b-qwen.gguf (distilled from the Qwen 27B teacher, current-state input only) and falls back to
/// model.gguf. Defaults can be overridden with Documents/players.json, for example
/// {"finetuned": "pacman-0.8b-v2.gguf", "phi4-mini-context": 2048}.
struct PlayerConfig: Decodable {
    var finetuned: String?
    var phi4Mini: String?
    var phi4MiniContext: UInt32?

    enum CodingKeys: String, CodingKey {
        case finetuned
        case phi4Mini = "phi4-mini"
        case phi4MiniContext = "phi4-mini-context"
    }

    static var url: URL { EngineHost.documents.appendingPathComponent("players.json") }

    static func load() -> PlayerConfig {
        guard let data = try? Data(contentsOf: url),
              let config = try? JSONDecoder().decode(PlayerConfig.self, from: data) else { return PlayerConfig() }
        return config
    }

    /// The model file name in Documents. Only the last path component is used.
    func modelFile(for player: Player) -> String? {
        let name: String?
        switch player {
        case .finetuned: name = finetuned ?? Self.defaultFinetuned
        case .phi4Mini: name = phi4Mini ?? "phi4-mini.gguf"
        case .jev: name = nil
        }
        return name.map { URL(fileURLWithPath: $0).lastPathComponent }
    }

    /// The distilled student if it is on the device, otherwise the earlier oracle-trained model.
    static var defaultFinetuned: String {
        let distilled = "pacman-0.8b-qwen.gguf"
        let path = EngineHost.documents.appendingPathComponent(distilled).path
        return FileManager.default.fileExists(atPath: path) ? distilled : "model.gguf"
    }

    var chatContextLength: UInt32 {
        phi4MiniContext.map { min(max($0, 1024), 8192) } ?? ChatMoveEngine.defaultContextLength
    }
}
