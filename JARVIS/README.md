# Jarvis Local Desktop Assistant

Jarvis is a local voice-first desktop assistant for Windows. It talks through the terminal, uses LM Studio on `localhost`, and can safely open apps, close visible windows, open websites, and search Google.

## Features

- Voice-first terminal assistant with optional type mode.
- Understands Turkish or English input, answers in English.
- Developer mode for prompts, raw model output, and token usage when LM Studio reports it.
- Safe app launcher using Start Menu/Desktop/Programs indexes.
- Explicit memory only: Jarvis remembers notes only when you say `remember that ...`.
- Extra tools: `/help` and `/apps <name>` in type mode.

## Safety

Jarvis does not run arbitrary shell commands from the model. AI output is restricted to safe actions like:

- `open_app:<name>`
- `open_web:<site-or-url>`
- `search_google:<query>`
- `close_app:<window-name>`
- `open_cmd`
- `chat`

Blocked intents include install, uninstall, delete, remove, update, download, edit, modify, registry, PowerShell, terminal, scripts, screenshots, recordings, and email.

## Setup

1. Start LM Studio.
2. Load `google/gemma-3-4b`.
3. Start the LM Studio local server at:

```text
http://localhost:1234/v1
```

4. Install Python dependencies if needed:

```powershell
python -m pip install -r requirements.txt
```

## Run

Voice mode:

```powershell
python .\jarvis.py
```

Type mode:

```powershell
python .\jarvis.py --type
```

Or double-click:

```text
start_jarvis_agent.bat
```

## Useful Commands

- `developer mode` or `development mode` - show prompts/raw outputs/token usage.
- `normal mode` - hide developer details.
- `remember that my favorite editor is VS Code` - save an explicit memory note.
- `clear memory` - clear saved notes.
- `/apps code` - list matching indexed apps in type mode.
- `/help` - show command help.

## Notes

The microphone feature uses Google speech recognition through `SpeechRecognition`. This is the one privacy tradeoff in the current version. LM Studio model calls stay on localhost.
