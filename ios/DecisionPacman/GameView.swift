import SwiftUI
import WebKit

/// The web game from the repository root, bundled under Web/ and served from
/// app://local/ so its ES modules load, with decisions answered on device or, for Jev, by native code
/// that holds the API key.
struct GameView: UIViewRepresentable {
    let initialPlayer: Player

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.setURLSchemeHandler(BundleSchemeHandler(), forURLScheme: "app")
        config.userContentController.addScriptMessageHandler(DecideHandler(), contentWorld: .page, name: "decide")
        config.userContentController.addScriptMessageHandler(JevHandler(), contentWorld: .page, name: "jev")
        config.userContentController.add(StatusHandler(), name: "status")
        // Let the game's Web Audio music start without waiting for a tap.
        config.mediaTypesRequiringUserActionForPlayback = []
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.isOpaque = false
        webView.backgroundColor = .black
        webView.scrollView.backgroundColor = .black
        webView.isInspectable = true
        webView.load(URLRequest(url: URL(string: "app://local/index.html?model=\(initialPlayer.rawValue)")!))
        WebViewHolder.shared.webView = webView
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}
}

/// Keeps a reference to the game's web view so URL commands can reach the page.
@MainActor
final class WebViewHolder {
    static let shared = WebViewHolder()
    weak var webView: WKWebView?

    /// Tells the page which player to use.
    func setPlayer(_ player: Player) {
        webView?.evaluateJavaScript("window.__pacman && window.__pacman.setModel('\(player.rawValue)')")
    }

    /// decisionpacman://control?model=finetuned|phi4-mini|jev&mode=ai|human&paused=0|1&speed=<x>&restart=1
    /// switches the player and calls the page's control functions.
    func handle(url: URL) {
        guard url.scheme == "decisionpacman", url.host == "control",
              let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems else { return }
        for item in items {
            guard let value = item.value else { continue }
            var js: String?
            switch item.name {
            case "model": if let player = Player(rawValue: value) { PlayerStore.shared.select(player) }
            case "mode" where value == "ai" || value == "human": js = "window.__pacman.setMode('\(value)')"
            case "paused": js = "window.__pacman.setPaused(\(value == "1"))"
            case "speed": if let x = Double(value), x > 0, x <= 2 { js = "window.__pacman.setSpeed(\(x))" }
            case "restart" where value == "1": js = "window.__pacman.restart()"
            default: break
            }
            if let js { webView?.evaluateJavaScript(js) }
        }
    }
}

/// Answers the page's on-device requests, `window.webkit.messageHandlers.decide.postMessage(...)`:
/// {player, prompt, options} for the fine-tuned decision model, {player, prompt, grammar, keys} for a chat model.
final class DecideHandler: NSObject, WKScriptMessageHandlerWithReply {
    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage,
                               replyHandler: @escaping @MainActor @Sendable (Any?, String?) -> Void) {
        let respond: @Sendable ([String: Any]) -> Void = { result in
            let payload = NSDictionary(dictionary: result)
            Task { @MainActor in replyHandler(payload, nil) }
        }
        guard let body = message.body as? [String: Any], let prompt = body["prompt"] as? String else {
            return respond(["error": "bad request"])
        }
        let player = (body["player"] as? String).flatMap(Player.init(rawValue:)) ?? .finetuned
        if let grammar = body["grammar"] as? String, let keys = body["keys"] as? [String], keys.count >= 2 {
            EngineHost.shared.generate(player: player, prompt: prompt, grammar: grammar, keys: keys, reply: respond)
        } else if let options = (body["options"] as? NSNumber)?.intValue, options >= 2 {
            EngineHost.shared.decide(player: player, prompt: prompt, options: options, reply: respond)
        } else {
            respond(["error": "bad request"])
        }
    }
}

/// Forwards the page's Jev requests, `window.webkit.messageHandlers.jev.postMessage({body})`, to the hosted API.
final class JevHandler: NSObject, WKScriptMessageHandlerWithReply {
    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage,
                               replyHandler: @escaping @MainActor @Sendable (Any?, String?) -> Void) {
        guard let body = (message.body as? [String: Any])?["body"] as? String else {
            replyHandler(["error": "bad request"], nil)
            return
        }
        Task {
            let result = await JevClient.send(body: body)
            let payload = NSDictionary(dictionary: result)
            await MainActor.run { replyHandler(payload, nil) }
        }
    }
}

/// Writes the page's periodic status to Documents/game_status.json, so a run can be checked from a Mac with devicectl.
final class StatusHandler: NSObject, WKScriptMessageHandler {
    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        guard var status = message.body as? [String: Any] else { return }
        // Sustained inference can heat the device, which slows decisions; record it next to the latency.
        let thermal = ["nominal", "fair", "serious", "critical"]
        status["thermal"] = thermal[min(ProcessInfo.processInfo.thermalState.rawValue, thermal.count - 1)]
        // The native side of the active player: model file and load state (never the Jev key).
        status["engine"] = EngineHost.shared.status.dictionary
        guard JSONSerialization.isValidJSONObject(status),
              let data = try? JSONSerialization.data(withJSONObject: status, options: [.sortedKeys]) else { return }
        try? data.write(to: EngineHost.documents.appendingPathComponent("game_status.json"), options: .atomic)
    }
}

/// Serves files from the app bundle's Web folder for the app:// scheme.
final class BundleSchemeHandler: NSObject, WKURLSchemeHandler {
    private let root = Bundle.main.url(forResource: "Web", withExtension: nil)!.standardizedFileURL

    func webView(_ webView: WKWebView, start task: WKURLSchemeTask) {
        guard let url = task.request.url else { return }
        var path = url.path
        if path.isEmpty || path == "/" { path = "/index.html" }
        let file = root.appendingPathComponent(String(path.dropFirst())).standardizedFileURL
        guard file.path.hasPrefix(root.path), let data = try? Data(contentsOf: file) else {
            task.didReceive(HTTPURLResponse(url: url, statusCode: 404, httpVersion: "HTTP/1.1", headerFields: nil)!)
            task.didFinish()
            return
        }
        let types = ["html": "text/html", "js": "text/javascript", "css": "text/css", "json": "application/json",
                     "svg": "image/svg+xml", "png": "image/png"]
        let mime = types[file.pathExtension.lowercased()] ?? "application/octet-stream"
        task.didReceive(HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1",
                                        headerFields: ["Content-Type": mime, "Content-Length": "\(data.count)"])!)
        task.didReceive(data)
        task.didFinish()
    }

    func webView(_ webView: WKWebView, stop task: WKURLSchemeTask) {}
}
