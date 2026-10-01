import Foundation
import llama

struct ChatMoveResult {
    /// The JSON answer, for example {"move": "left"}. Text after the move is forced by the grammar and filled in.
    let text: String
    let move: String
    /// Option probabilities at the move token, in the order of the keys, or nil if none of the candidates matched.
    let probabilities: [Double]?
    let promptTokens: Int
    let outputTokens: Int
    let prefillMilliseconds: Double
    let milliseconds: Double
}

/// Runs a plain chat model the way the repository's `llm:` policy runs it through Ollama: the rendered
/// chat prompt, greedy decoding (temperature 0) under a GBNF grammar that only admits
/// {"move": "<one of the options>"}, and option probabilities read from the logits at the move token,
/// renormalized over the options. Generation stops as soon as the move is decided, since the grammar
/// forces everything after it.
final class ChatMoveEngine {
    private let model: OpaquePointer
    private let context: OpaquePointer
    private let vocab: OpaquePointer
    private let vocabSize: Int
    private let batchSize: Int
    private var candidates: [String: [llama_token]] = [:]
    private static let maxNewTokens = 32

    /// The context length matters for phi-family models with LongRoPE (phi4-mini): llama.cpp uses the long
    /// rope factors when the context exceeds the model's original 4096 tokens, the short ones otherwise.
    /// Ollama on a large Mac runs with a longer context, so the evaluated phi4-mini used the long factors;
    /// the default of 4352 reproduces its answers. 2048 uses the short factors and about 300 MB less memory.
    static let defaultContextLength: UInt32 = 4352

    init(modelPath: String, contextLength: UInt32 = ChatMoveEngine.defaultContextLength, batchSize: UInt32 = 512) throws {
        llama_backend_init()
        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = 99
        guard let model = llama_model_load_from_file(modelPath, modelParams) else {
            throw DecisionEngineError.loadFailed(modelPath)
        }
        var contextParams = llama_context_default_params()
        contextParams.n_ctx = contextLength
        contextParams.n_batch = batchSize
        contextParams.n_ubatch = batchSize
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
        self.vocabSize = Int(llama_vocab_n_tokens(vocab))
        self.batchSize = Int(batchSize)
    }

    deinit {
        llama_free(context)
        llama_model_free(model)
    }

    private func piece(_ token: llama_token) -> String {
        var buf = [CChar](repeating: 0, count: 64)
        var n = llama_token_to_piece(vocab, token, &buf, Int32(buf.count), 0, false)
        if n < 0 {
            buf = [CChar](repeating: 0, count: Int(-n))
            n = llama_token_to_piece(vocab, token, &buf, Int32(buf.count), 0, false)
        }
        return String(decoding: buf.prefix(Int(max(0, n))).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// Tokens that can start the option's value: their text, without leading whitespace and quotes,
    /// is a non-empty prefix of the option (the same rule as `moveProbabilities` in src/agent/llm.ts).
    private func candidateTokens(for key: String) -> [llama_token] {
        if let cached = candidates[key] { return cached }
        var ids: [llama_token] = []
        for id in 0..<vocabSize {
            let text = piece(llama_token(id)).drop(while: { $0 == "\"" || $0.isWhitespace }).lowercased()
            if !text.isEmpty && key.hasPrefix(text) { ids.append(llama_token(id)) }
        }
        candidates[key] = ids
        return ids
    }

    private func decode(_ tokens: [llama_token], from start: Int) throws {
        var offset = 0
        while offset < tokens.count {
            let chunk = tokens[offset..<min(tokens.count, offset + batchSize)]
            var batch = llama_batch_init(Int32(chunk.count), 0, 1)
            defer { llama_batch_free(batch) }
            for (i, token) in chunk.enumerated() {
                batch.token[i] = token
                batch.pos[i] = llama_pos(start + offset + i)
                batch.n_seq_id[i] = 1
                batch.seq_id[i]![0] = 0
                batch.logits[i] = offset + i == tokens.count - 1 ? 1 : 0
            }
            batch.n_tokens = Int32(chunk.count)
            let status = llama_decode(context, batch)
            guard status == 0 else { throw DecisionEngineError.decodeFailed(status) }
            offset += chunk.count
        }
    }

    /// Greedy choice among the tokens the grammar allows. Tries the unconstrained best token first, which
    /// is almost always allowed, so the full-vocabulary grammar pass runs only when it is not.
    private func pick(logits: UnsafeMutablePointer<Float>, grammar: UnsafeMutablePointer<llama_sampler>) -> llama_token {
        var best = 0
        for i in 1..<vocabSize where logits[i] > logits[best] { best = i }
        var single = [llama_token_data(id: llama_token(best), logit: logits[best], p: 0)]
        let allowed = single.withUnsafeMutableBufferPointer { buf -> Bool in
            var arr = llama_token_data_array(data: buf.baseAddress, size: 1, selected: -1, sorted: false)
            llama_sampler_apply(grammar, &arr)
            return buf[0].logit.isFinite
        }
        if allowed { return llama_token(best) }
        var all = (0..<vocabSize).map { llama_token_data(id: llama_token($0), logit: logits[$0], p: 0) }
        return all.withUnsafeMutableBufferPointer { buf -> llama_token in
            var arr = llama_token_data_array(data: buf.baseAddress, size: buf.count, selected: -1, sorted: false)
            llama_sampler_apply(grammar, &arr)
            var choice = 0
            for i in 0..<arr.size where arr.data[i].logit > arr.data[choice].logit { choice = i }
            return arr.data[choice].id
        }
    }

    /// Where the move's value starts in `text` (just after its opening quote), if it has started.
    private static func valueStart(in text: String) -> String.Index? {
        guard let key = text.range(of: "\"move\"") else { return nil }
        var i = key.upperBound
        while i < text.endIndex, text[i].isWhitespace { i = text.index(after: i) }
        guard i < text.endIndex, text[i] == ":" else { return nil }
        i = text.index(after: i)
        while i < text.endIndex, text[i].isWhitespace { i = text.index(after: i) }
        guard i < text.endIndex, text[i] == "\"" else { return nil }
        return text.index(after: i)
    }

    /// The move once only one option is consistent with the text generated so far.
    private static func decidedMove(in text: String, keys: [String]) -> String? {
        guard let start = valueStart(in: text) else { return nil }
        let value = text[start...]
        if let close = value.firstIndex(of: "\"") {
            let exact = String(value[..<close])
            return keys.contains(exact) ? exact : nil
        }
        guard !value.isEmpty else { return nil }
        let matches = keys.filter { $0.hasPrefix(value) }
        return matches.count == 1 ? matches[0] : nil
    }

    func generate(prompt: String, grammar gbnf: String, keys: [String]) throws -> ChatMoveResult {
        let start = DispatchTime.now()
        let elapsed = { Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000 }
        guard let grammar = llama_sampler_init_grammar(vocab, gbnf, "root") else { throw DecisionEngineError.grammarFailed }
        defer { llama_sampler_free(grammar) }

        // add_special follows the model's own BOS setting, as Ollama does.
        let tokens = try DecisionEngine.tokenize(vocab: vocab, text: prompt, parseSpecial: true, addSpecial: true)
        llama_memory_clear(llama_get_memory(context), true)
        try decode(tokens, from: 0)
        llama_synchronize(context)  // Metal runs the batch asynchronously; wait so prefill time is real.
        let prefill = elapsed()

        var text = ""
        var probabilities: [Double]?
        var produced = 0
        var move: String?
        for _ in 0..<Self.maxNewTokens {
            let logits = llama_get_logits_ith(context, -1)!
            let token = pick(logits: logits, grammar: grammar)
            if llama_vocab_is_eog(vocab, token) { break }
            let next = text + piece(token)
            if probabilities == nil, let vs = Self.valueStart(in: next), vs < next.endIndex,
               Self.valueStart(in: text).map({ $0 >= text.endIndex }) ?? true {
                probabilities = optionProbabilities(logits: logits, keys: keys)
            }
            llama_sampler_accept(grammar, token)
            text = next
            produced += 1
            if let decided = Self.decidedMove(in: text, keys: keys) {
                move = decided
                break
            }
            try decode([token], from: tokens.count + produced - 1)
        }
        guard let move, let vs = Self.valueStart(in: text) else { throw DecisionEngineError.noAnswer(text) }
        let answer = String(text[..<vs]) + move + "\"}"
        return ChatMoveResult(text: answer, move: move, probabilities: probabilities, promptTokens: tokens.count,
                              outputTokens: produced, prefillMilliseconds: prefill, milliseconds: elapsed())
    }

    private func optionProbabilities(logits: UnsafeMutablePointer<Float>, keys: [String]) -> [Double]? {
        let ids = keys.map { candidateTokens(for: $0) }
        let peak = ids.flatMap { $0 }.map { Double(logits[Int($0)]) }.max() ?? 0
        let mass = ids.map { $0.reduce(0.0) { $0 + exp(Double(logits[Int($1)]) - peak) } }
        let total = mass.reduce(0, +)
        return total > 0 ? mass.map { $0 / total } : nil
    }
}
