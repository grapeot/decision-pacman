import Foundation
import llama

enum DecisionEngineError: LocalizedError {
    case loadFailed(String)
    case contextFailed
    case decodeFailed(Int32)
    case tokenizeFailed
    case grammarFailed
    case noAnswer(String)

    var errorDescription: String? {
        switch self {
        case .loadFailed(let path): return "Could not load model at \(path)"
        case .contextFailed: return "Could not create an inference context"
        case .decodeFailed(let code): return "llama_decode failed with code \(code)"
        case .tokenizeFailed: return "Tokenization failed"
        case .grammarFailed: return "Could not parse the answer grammar"
        case .noAnswer(let text): return "The model gave no complete move: \(text.prefix(80))"
        }
    }
}

struct DecisionResult {
    let probabilities: [Double]
    let tokens: Int
    let milliseconds: Double
}

/// Scores decision prompts the way Ollama's /v1/systemone does: run the whole
/// prompt once, read the logits of the option letters at the next position,
/// and take a softmax over those letters only.
final class DecisionEngine {
    private let model: OpaquePointer
    private let context: OpaquePointer
    private let vocab: OpaquePointer
    private let letterTokens: [llama_token]

    init(modelPath: String, contextLength: UInt32 = 1024) throws {
        llama_backend_init()
        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = 99
        guard let model = llama_model_load_from_file(modelPath, modelParams) else {
            throw DecisionEngineError.loadFailed(modelPath)
        }
        var contextParams = llama_context_default_params()
        contextParams.n_ctx = contextLength
        contextParams.n_batch = contextLength
        contextParams.n_ubatch = contextLength
        let threads = Int32(max(1, min(8, ProcessInfo.processInfo.activeProcessorCount - 2)))
        contextParams.n_threads = threads
        contextParams.n_threads_batch = threads
        guard let context = llama_init_from_model(model, contextParams) else {
            llama_model_free(model)
            throw DecisionEngineError.contextFailed
        }
        self.model = model
        self.context = context
        self.vocab = llama_model_get_vocab(model)
        var letters: [llama_token] = []
        for scalar in "ABCDEFGHIJKLMNOPQRSTUVWXYZ".unicodeScalars {
            let ids = try DecisionEngine.tokenize(vocab: vocab, text: String(scalar), parseSpecial: false)
            letters.append(ids[0])
        }
        self.letterTokens = letters
    }

    deinit {
        llama_free(context)
        llama_model_free(model)
    }

    static func tokenize(vocab: OpaquePointer, text: String, parseSpecial: Bool, addSpecial: Bool = false) throws -> [llama_token] {
        let utf8Count = Int32(text.utf8.count)
        var capacity = utf8Count + 16
        var tokens = [llama_token](repeating: 0, count: Int(capacity))
        var count = llama_tokenize(vocab, text, utf8Count, &tokens, capacity, addSpecial, parseSpecial)
        if count < 0 {
            capacity = -count
            tokens = [llama_token](repeating: 0, count: Int(capacity))
            count = llama_tokenize(vocab, text, utf8Count, &tokens, capacity, addSpecial, parseSpecial)
        }
        guard count > 0 else { throw DecisionEngineError.tokenizeFailed }
        return Array(tokens.prefix(Int(count)))
    }

    /// Scores one rendered prompt (chat template already applied) over `optionCount` lettered options.
    func decide(prompt: String, optionCount: Int) throws -> DecisionResult {
        let start = DispatchTime.now()
        let tokens = try DecisionEngine.tokenize(vocab: vocab, text: prompt, parseSpecial: true)
        // Clear both the attention cache and the recurrent (Gated DeltaNet) state.
        llama_memory_clear(llama_get_memory(context), true)

        var batch = llama_batch_init(Int32(tokens.count), 0, 1)
        defer { llama_batch_free(batch) }
        for (i, token) in tokens.enumerated() {
            batch.token[i] = token
            batch.pos[i] = llama_pos(i)
            batch.n_seq_id[i] = 1
            batch.seq_id[i]![0] = 0
            batch.logits[i] = i == tokens.count - 1 ? 1 : 0
        }
        batch.n_tokens = Int32(tokens.count)
        let status = llama_decode(context, batch)
        guard status == 0 else { throw DecisionEngineError.decodeFailed(status) }

        let logits = llama_get_logits_ith(context, -1)!
        let scores = letterTokens.prefix(optionCount).map { Double(logits[Int($0)]) }
        let peak = scores.max() ?? 0
        let exps = scores.map { exp($0 - peak) }
        let total = exps.reduce(0, +)
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
        return DecisionResult(probabilities: exps.map { $0 / total }, tokens: tokens.count, milliseconds: elapsed)
    }
}
