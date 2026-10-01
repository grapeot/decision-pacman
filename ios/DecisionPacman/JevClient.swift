import Foundation

/// Calls TypeSafe's hosted Jev for the page. The page cannot call it from app://local (no CORS for that
/// origin), and the key must not reach the page: it is read from Documents/jev_key.txt for every request
/// and only ever placed in the Authorization header. The endpoint is fixed, so the page cannot send the
/// key anywhere else. Nothing here logs the key or the request.
enum JevClient {
    static let endpoint = URL(string: "https://api.typesafe.ai/v1/systemone")!
    static var keyURL: URL { EngineHost.documents.appendingPathComponent("jev_key.txt") }

    private static let session: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 15
        config.urlCache = nil
        return URLSession(configuration: config)
    }()

    private static func readKey() -> String? {
        guard let text = try? String(contentsOf: keyURL, encoding: .utf8) else { return nil }
        let key = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return key.isEmpty ? nil : key
    }

    static var hasKey: Bool { readKey() != nil }

    /// Posts a /v1/systemone request body. Replies with {status, body} from the API, or {error} when there is
    /// no key or no connection.
    static func send(body: String) async -> [String: Any] {
        guard let key = readKey() else {
            return ["error": "No Jev API key: copy it into the app's Documents folder as jev_key.txt."]
        }
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        request.httpBody = Data(body.utf8)
        do {
            let (data, response) = try await session.data(for: request)
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            return ["status": status, "body": String(decoding: data, as: UTF8.self)]
        } catch {
            return ["error": "Cannot reach Jev: \((error as? URLError)?.code.rawValue ?? 0) \(error.localizedDescription)"]
        }
    }
}
