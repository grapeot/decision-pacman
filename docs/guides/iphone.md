# iPhone app

The iPhone app (`ios/`) is a SwiftUI app that serves the same web build in a WKWebView and answers its decisions natively. A segmented control above the maze picks one of three players. All three see the same current-state input as the players on the [ladder](../results.md).

| Picker label | Where | How it decides |
|---|---|---|
| 0.8B tuned | On device | The distilled `pacman-0.8b-qwen` (or another fine-tuned decision model): one forward pass over the System One prompt, softmax over the option letters (llama.cpp on Metal). |
| phi4-mini | On device | A plain 3.8B chat model asked like `llm:phi4-mini`: same facts and JSON answer line, temperature 0, under a GBNF grammar that only admits `{"move": "<option>"}`. |
| Jev | Cloud | TypeSafe's hosted Jev; native code sends the request to a fixed endpoint, so the key never reaches the page. |

## What you need

- A Mac with Xcode and `xcodegen`.
- An iPhone paired with the Mac.
- A development provisioning profile for your team.
- The GGUF model files. The distilled `pacman-0.8b-qwen` is on [Hugging Face](https://huggingface.co/grapeot/decision-pacman-0.8b-GGUF):
  ```sh
  hf download grapeot/decision-pacman-0.8b-GGUF pacman-0.8b-qwen-Q8_0.gguf --local-dir .
  ```
  phi4-mini comes from Ollama (`ollama pull phi4-mini`).
- A TypeSafe API key from your password manager for Jev.

The llama.cpp xcframework is downloaded at a pinned tag. It has no simulator slice, so the app runs on a device only.

## Build and install

List paired devices to find device names and identifiers:

```sh
xcrun devicectl list devices
```

Build and install the app on a paired device:

```sh
ios/scripts/build_device.sh <device-name>
```

This script downloads the pinned llama.cpp xcframework (`ios/scripts/fetch_llama.sh`, tag b11298), builds the web game into the app (`ios/scripts/build_web.sh`), runs `xcodegen generate`, reads the team from the cached profile so it never appears in the repo, builds, and installs. It looks for your team's cached wildcard development profile, named `iOS Team Provisioning Profile: *`; set `PROFILE_NAME` to use another one.

To verify compilation without signing (CI-style check), run `ios/scripts/build_web.sh` and `xcodegen generate` in `ios/`, then run:

```sh
xcodebuild -project ios/DecisionPacman.xcodeproj -scheme DecisionPacman -destination "generic/platform=iOS" CODE_SIGNING_ALLOWED=NO build
```

## Copy models

Models and the Jev key are copied into the app's Documents folder, never bundled, and never committed. The bundle identifier is `io.github.grapeot.decisionpacman`.

Set your device identifier and copy the model files using the copy helper:

```sh
dev=<device-identifier>
copy() { xcrun devicectl device copy to --device "$dev" --domain-type appDataContainer \
  --domain-identifier io.github.grapeot.decisionpacman --source "$1" --destination "Documents/$2"; }
copy pacman-0.8b-qwen-Q8_0.gguf pacman-0.8b-qwen.gguf   # distilled 0.8B from Hugging Face (Q8_0, 795 MB)
copy "$(ollama show phi4-mini --modelfile | awk '/^FROM /{print $2}')" phi4-mini.gguf   # Q4_K_M, about 2.3 GB
```

The fine-tuned player loads `pacman-0.8b-qwen.gguf` from Documents and falls back to `model.gguf`, an older file name. phi4-mini loads `phi4-mini.gguf`. If you trained the model yourself ([distill.md](distill.md)), copy `runs/ft_q1/pacman-0.8b-qwen-q8_0.gguf` instead.

You can override default file names with `Documents/players.json`, for example `{"finetuned": "pacman-0.8b-v2.gguf"}`. The same file can name another phi4-mini file (`"phi4-mini"`) or set `"phi4-mini-context": 2048`. Setting context to 2048 saves about 300 MB, but changes phi4-mini's answers: with a 2048-token context it agreed with Ollama on 33 of 40 moves, whereas 4352 or 8192 gave 40 of 40.

## Add the Jev key

Obtain your key from your password manager. Write it into a temporary file, copy it to the device, and delete the temporary file immediately. Never paste it into a file you keep:

```sh
tmp="$(mktemp)"
# write the key into "$tmp" from your password manager's command-line tool
copy "$tmp" jev_key.txt
rm -f "$tmp"
```

The app reads `Documents/jev_key.txt`. The key never enters the page, the bundle, the repository, or the status file.

## Drive the app from the Mac

You can switch players with the picker above the maze, or from the Mac while the app runs using the openURL control route:

```sh
xcrun devicectl device process openURL --device "$dev" "decisionpacman://control?model=phi4-mini"
```

Accepted values for `model` are `finetuned`, `phi4-mini`, and `jev`. The control route also takes `mode=ai|human`, `paused=0|1`, `speed=<x>`, `wait=0|1`, and `restart=1`.

The line under the picker shows the model file and whether it is loaded. The Decision panel shows the active player and its latency. Switching frees the previous on-device model before loading the next; only one model is in memory. The choice is remembered across launches.

When measuring, launch the app in the foreground:

```sh
xcrun devicectl device process launch --device "$dev" --activate io.github.grapeot.decisionpacman
```

Without `--activate`, the app can sit behind other UI and its timers stop, which looks like a frozen game. Check that the heartbeat's frame count is non-zero first.

Every 2 s the page writes `Documents/game_status.json` with the score, the active player (`model`, `modelName`), the native load state (`engine`), decision latency, frame timing, and the thermal state. Copy it to the Mac to read it:

```sh
xcrun devicectl device copy from --device "$dev" --domain-type appDataContainer \
  --domain-identifier io.github.grapeot.decisionpacman --source Documents/game_status.json --destination game_status.json
```

## Check inference on a Mac

Check the app's inference code on a Mac against fixed prompts and Ollama's answers:

```sh
ios/scripts/mac_check.sh decision pacman-0.8b-qwen-Q8_0.gguf
ios/scripts/mac_check.sh chat "$(ollama show phi4-mini --modelfile | awk '/^FROM /{print $2}')"
```

The decision check uses `ios/mac_check/bench_prompts.json`: 20 rendered game prompts with the option probabilities Ollama returns for `pacman-0.8b-qwen`. With the published GGUF, the app's Swift engine on an M3 Ultra chose the same option on 60 of 60 runs (20 prompts, 3 rounds), with a largest probability difference of 0.0004, at 34 ms per decision.

Measured on a Mac (M3 Ultra, llama.cpp b11298, Metal):
- Fine-tuned 0.8B Q8_0 (the earlier `pacman-0.8b`): same choice as Ollama on 60 of 60, max probability difference 0.0003, p50 36 ms.
- phi4-mini Q4_K_M: 40 of 40 legal and the same move as Ollama, p50 168-173 ms.

## Performance on the phone

On an iPhone 16 Pro Max at 1x speed (2026-10-01):

| Player | Decision latency | Notes |
|---|---|---|
| Distilled 0.8B (`pacman-0.8b-qwen`) | about 400 ms (p50 393-425 ms over the first 18 decisions) | |
| phi4-mini | 2.2-2.9 s | Loads in 4.9 s; no memory trouble |
| Jev | about 125-155 ms | Over Wi-Fi, key read from Documents |

A 0.8B decision takes about 400 ms on the phone against 55-63 ms on the M3 Ultra. In the first phone test (with an earlier 0.8B fine-tune of the same size), 27% of answers arrived after Pac-Man had passed their junction, against 13% on the Mac. The same earlier tests showed that latency also climbs with heat: from about 360-375 ms at the start of a game to 650-850 ms after 40-60 s of continuous inference (thermal state `serious`), with 23-45% stale answers per game. The game itself held 30 ticks per second during inference. The first model load took 17 s (shader compilation); later loads took 0.37 s. phi4-mini takes seconds per answer. A slower game alone does not fix that: at 0.25x, phi4-mini still answered in 2.5-2.6 s per decision (thermal state `serious`), and 8 of its first 18 decisions arrived after Pac-Man had passed the junction.

A later screen recording, [`docs/media/iphone_pacman_08b_qwen.mp4`](../media/iphone_pacman_08b_qwen.mp4) (29 s, 1x), shows the distilled `pacman-0.8b-qwen` at about 505 ms per decision with 25-28% late answers. It ate 174 pellets for 2,500 points and lost no lives.

Only one on-device model is loaded at a time; switching frees the previous model before loading the next. phi4-mini takes about 3.3 GB with its cache.

## Use Wait for phi4-mini

Turn on the Wait switch for a phi4-mini demo: the checkbox in the controls, or from the Mac:

```sh
xcrun devicectl device process openURL --device "$dev" "decisionpacman://control?wait=1"
```

The model is asked about the next junction while the game moves. If Pac-Man is about to reach that junction before the answer arrives, the whole game holds there, ghosts and timers included, until it does. A slow model is then judged on its choices, not its speed. On the iPhone 16 Pro Max at 1x with Wait on, phi4-mini made 26 decisions, none late, at about 2.2 s each; the game held in the windows where it waited. The status file reports the switch as `waitForModel`. With Wait on, latency no longer costs game time, so the play is not comparable with the realtime ladder in [results.md](../results.md).

The app uses an ambient audio session, so Silent Mode mutes the game.
