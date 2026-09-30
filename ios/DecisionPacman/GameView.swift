import SwiftUI
import WebKit

/// The web game from the repository root, bundled under Web/ and served from
/// app://local/ so its ES modules load, with decisions answered on device.
struct GameView: UIViewRepresentable {
    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.setURLSchemeHandler(BundleSchemeHandler(), forURLScheme: "app")
        config.userContentController.addScriptMessageHandler(DecideHandler(), contentWorld: .page, name: "decide")
        config.userContentController.add(StatusHandler(), name: "status")
        let webView = WKWebView(frame: .zero, configuration: config)
        webView.isOpaque = false
        webView.backgroundColor = .black
        webView.scrollView.backgroundColor = .black
        webView.isInspectable = true
        webView.load(URLRequest(url: URL(string: "app://local/index.html")!))
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}
}

/// Answers the page's `window.webkit.messageHandlers.decide.postMessage({prompt, options})`.
final class DecideHandler: NSObject, WKScriptMessageHandlerWithReply {
    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage,
                               replyHandler: @escaping @MainActor @Sendable (Any?, String?) -> Void) {
        guard let body = message.body as? [String: Any], let prompt = body["prompt"] as? String,
              let options = (body["options"] as? NSNumber)?.intValue, options >= 2 else {
            replyHandler(["error": "bad request"], nil)
            return
        }
        EngineHost.shared.decide(prompt: prompt, options: options) { result in
            let payload = NSDictionary(dictionary: result)
            Task { @MainActor in replyHandler(payload, nil) }
        }
    }
}

/// Writes the page's periodic status to Documents/game_status.json, so a run can be checked from a Mac with devicectl.
final class StatusHandler: NSObject, WKScriptMessageHandler {
    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        guard JSONSerialization.isValidJSONObject(message.body),
              let data = try? JSONSerialization.data(withJSONObject: message.body, options: [.sortedKeys]) else { return }
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
