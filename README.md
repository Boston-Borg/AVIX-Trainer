# AVX — Your 24/7 PPL Study Companion

AI-powered Private Pilot License coaching: CFI chat, study guides, and a conversational DPE oral exam simulator.

This repo is set up so the frontend (`AVX1.2.html`) talks to a tiny Python/Flask backend (`server.py`) that holds your Anthropic API key. **The key never ships to the browser** — that's what makes this safe to deploy on Render.

---

## Project layout

```
AVX/
├── AVX1.2.html        ← your existing UI
├── server.py          ← Flask app + /api/chat endpoint that calls Claude
├── requirements.txt   ← Python dependencies
├── .env.example       ← copy this to .env and paste your real key
├── .env               ← (you create this — gitignored)
├── .gitignore         ← keeps secrets and big PDFs out of git
├── render.yaml        ← Render Blueprint for one-click deploy
├── .vscode/           ← VS Code settings, recommended extensions, debugger
├── knowledge/         ← FAA reference PDFs (gitignored, see knowledge/README.md)
└── README.md          ← you are here
```

### A note on the `knowledge/` folder

Claude is already trained — we don't fine-tune it on the PDFs. Instead they're used as **reference context** at query time (RAG: retrieval-augmented generation). The PDFs are gitignored because they total ~140 MB and would bloat every Render deploy. See `knowledge/README.md` for the full list of sources and the RAG plan.

---

## 1. Open the project in VS Code

```bash
cd "/Users/bostonborg/Documents/Claude/Projects/AVX"
code .
```

When VS Code opens, it will offer to install the recommended extensions (Python, Pylance, debugpy). Accept.

---

## 2. Create a virtual environment + install dependencies

A virtual environment is just a private folder of Python packages so this project doesn't fight with other Python work on your machine.

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows PowerShell

pip install -r requirements.txt
```

VS Code should detect the `.venv` automatically. If it asks "Use this interpreter?", say yes.

---

## 3. Add your Claude API key

1. Get a key at https://console.anthropic.com/settings/keys
2. Make a copy of the template:

   ```bash
   cp .env.example .env
   ```

3. Open `.env` and replace `sk-ant-your-key-goes-here` with your real key.

`.env` is in `.gitignore`, so it will never be committed. Don't paste the key anywhere else.

---

## 4. Run it locally

```bash
python server.py
```

You should see something like `Running on http://0.0.0.0:8000`. Open http://localhost:8000 in your browser — the AVX page loads.

Sanity-check the API wiring:

```bash
curl http://localhost:8000/api/health
# → {"api_key_configured": true, "model": "claude-sonnet-4-6", "status": "ok"}
```

Try a real chat call:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is Vx?"}]}'
```

You should get a JSON response with a `reply` field.

---

## 5. Wire the frontend up to `/api/chat` (when you're ready)

In `AVX1.2.html`, wherever a feature currently uses a hard-coded mock response, replace it with a call like this:

```js
async function askClaude(messages, system) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, system })
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data.reply;
}
```

Then any of your three features (CFI chat, study guides, oral exam simulator) can call `askClaude(...)` with a different `system` prompt to give Claude the right persona.

---

## 6. Deploy to Render

1. Push this folder to a GitHub repo (make sure `.env` is NOT committed — `.gitignore` handles this, but double-check with `git status`).
2. Go to https://dashboard.render.com → **New +** → **Blueprint** → connect your repo.
3. Render reads `render.yaml` and creates the service.
4. In the new service's **Environment** tab, add a secret:
   - Key: `ANTHROPIC_API_KEY`
   - Value: your real key
5. Trigger a deploy. Done.

The key lives in Render's encrypted environment store, never in your code or git history.

---

## Common issues

- **`api_key_configured: false`** — your `.env` file is missing or the variable name is wrong. It must be exactly `ANTHROPIC_API_KEY`.
- **`ModuleNotFoundError: flask`** — your virtual environment isn't activated. Run `source .venv/bin/activate` and try again.
- **CORS error in the browser console** — only happens if you serve the HTML from a different origin than the Flask app. The default setup serves both from the same origin, so this shouldn't come up.

---

## What's next

Once you're comfortable, good places to extend:

- Add a `/api/study-guide` endpoint that takes a topic and returns a structured guide.
- Add a `/api/oral-exam` endpoint that keeps a session ID and runs a multi-turn DPE simulation.
- Add basic rate limiting (e.g. `flask-limiter`) so a runaway client can't drain your Anthropic budget.
