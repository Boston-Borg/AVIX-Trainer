"""
AVX Flask server.

This is a tiny backend whose job is to:
  1. Serve your AVX HTML page.
  2. Hold your Claude API key (loaded from an environment variable, never
     hard-coded) and forward chat requests to Claude.

Why a backend at all?
  Putting an API key in JavaScript that runs in the browser means anyone who
  visits your site can open DevTools, copy the key, and rack up charges on
  your Anthropic account. The browser talks to THIS server, this server talks
  to Claude. The key never leaves the server.

Run locally:
    pip install -r requirements.txt
    cp .env.example .env          # then paste your real key into .env
    python server.py

Run in production (Render uses this command, see render.yaml):
    gunicorn server:app
"""

import os
import logging
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Load variables from a local .env file if present. On Render, env vars come
# from the dashboard instead — load_dotenv just silently does nothing there.
load_dotenv()

# --- Anthropic client setup -------------------------------------------------
# We import lazily-friendly: if the key is missing we still want the server to
# start (so you can load the page), but /api/chat will return a clear error.
from anthropic import Anthropic, APIError

# RAG retrieval — loads the FAA chunk index at import time.
import retrieval

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

# Build the client only if we have a key. Otherwise leave it None and let the
# endpoint return a friendly error.
client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# --- Flask app --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
HTML_FILE = "AVX1.2.html"  # the page you already have

app = Flask(__name__, static_folder=str(PROJECT_ROOT), static_url_path="")

# CORS: allow the browser to call /api/* from the same origin. If you ever
# host the HTML separately, add that origin to the list below.
CORS(app, resources={r"/api/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("avx")


# --- Routes -----------------------------------------------------------------
@app.route("/")
def index():
    """Serve the AVX HTML page."""
    return send_from_directory(PROJECT_ROOT, HTML_FILE)


@app.route("/api/health")
def health():
    """Quick check that the server is up and whether the API key is wired."""
    return jsonify(
        status="ok",
        api_key_configured=bool(ANTHROPIC_API_KEY),
        model=CLAUDE_MODEL,
        rag_index_loaded=retrieval.index_ready(),
        rag_chunks=len(retrieval._index.chunks),
    )


RAG_INSTRUCTIONS = """You answer questions for a Private Pilot License (PPL) student.

You have been given excerpts from official FAA publications below, labeled like [1], [2], etc.
Each excerpt has a citation tag (e.g. "PHAK p.234" or "14 CFR 60-109 pp.727-728").

Rules:
- Ground every factual claim in the provided excerpts when possible.
- When you use information from an excerpt, cite it inline like "(PHAK p.234)" or "(14 CFR 91.155)".
- If the excerpts don't actually answer the question, say so honestly and answer from general aviation knowledge — but flag it as "general knowledge" so the student knows it's not from a cited source.
- Don't invent regulation numbers or page citations. If you didn't see it in the excerpts, don't cite it.
- Keep answers focused and conversational; use **bold** for key terms.

FAA EXCERPTS:
{context}
"""


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Forward a chat request to Claude, with RAG context inserted.

    Expected JSON body:
        {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi!"},
                {"role": "user", "content": "what's a Vx climb?"}
            ],
            "system": "Optional system prompt (e.g. 'You are a CFI...')",
            "max_tokens": 1024,    // optional
            "use_rag": true        // optional, default true
        }

    Returns:
        { "reply": "<text>", "model": "<name>", "citations": ["PHAK p.234", ...] }
    """
    if client is None:
        return (
            jsonify(
                error=(
                    "ANTHROPIC_API_KEY is not set. Add it to your .env file "
                    "for local dev, or to the Render dashboard for production."
                )
            ),
            500,
        )

    data = request.get_json(silent=True) or {}
    messages = data.get("messages")
    base_system = data.get(
        "system",
        "You are AVX, an FAA-accurate CFI helping a student pilot study for the "
        "Private Pilot License (PPL) checkride. Be precise and concise."
    )
    max_tokens = int(data.get("max_tokens", 1024))
    use_rag = bool(data.get("use_rag", True))

    if not isinstance(messages, list) or not messages:
        return jsonify(error="`messages` must be a non-empty list."), 400

    # ---- RAG: retrieve relevant FAA chunks for the latest user turn -------
    citations: list[str] = []
    system_prompt = base_system
    if use_rag and retrieval.index_ready():
        # Build a query from the most recent user message (could be smarter
        # — e.g. summarize the whole thread — but a single-turn query is
        # fine for the PPL Q&A pattern we have today).
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        if last_user:
            try:
                hits = retrieval.retrieve(last_user, top_k=5)
                if hits:
                    context_block = retrieval.format_context(hits)
                    system_prompt = (
                        base_system
                        + "\n\n"
                        + RAG_INSTRUCTIONS.format(context=context_block)
                    )
                    citations = [h.citation() for h in hits]
                    log.info("RAG: %d hits for query %r", len(hits), last_user[:80])
                else:
                    log.info("RAG: no hits above threshold for query %r", last_user[:80])
            except Exception:  # noqa: BLE001
                # Retrieval failures should never break the chat — fall back
                # to plain Claude.
                log.exception("Retrieval failed; falling back to no-RAG")

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
        )
        reply_text = "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )
        return jsonify(reply=reply_text, model=response.model, citations=citations)

    except APIError as e:
        log.exception("Claude API error")
        return jsonify(error=f"Claude API error: {e}"), 502
    except Exception as e:  # noqa: BLE001
        log.exception("Unexpected server error")
        return jsonify(error=f"Server error: {e}"), 500


# --- /api/generate ----------------------------------------------------------
# Used by the Resources page to generate Study Guides, Flash Cards, and
# Quizzes on demand. All three reuse the RAG pipeline so the content is
# grounded in your FAA PDFs, with citations.
import json as _json  # local alias to avoid shadowing


def _strip_code_fence(s: str) -> str:
    """Remove ```json ... ``` wrappers Claude sometimes adds around JSON."""
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]  # drop the first line (```json or ```)
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


_PROMPTS = {
    "study": (
        "Generate a comprehensive PPL study guide for the topic: \"{topic}\".\n\n"
        "Format as Markdown with these sections:\n"
        "  ## Overview  — 2-3 sentence summary\n"
        "  ## Key Concepts  — bulleted, with **bold** key terms\n"
        "  ## Common Pitfalls  — what students get wrong on the checkride\n"
        "  ## Sample Checkride Questions  — 3 questions with brief answers\n\n"
        "Cite FAA sources inline like (PHAK p.234) or (14 CFR 91.155). "
        "Be precise — if the excerpts don't cover something, don't guess."
    ),
    "flashcards": (
        "Generate {count} flash cards for the topic: \"{topic}\".\n\n"
        "Output ONLY a JSON array. No commentary, no markdown fences. "
        "Each card has exactly two keys:\n"
        '  {{"q": "<question>", "a": "<answer with FAA citation inline>"}}\n\n'
        "Make questions varied: definitions, numbers, procedures, scenarios. "
        "Keep answers tight — 1-3 sentences. Always cite the source like (PHAK p.234). "
        "Generate exactly {count} cards — no more, no fewer."
    ),
    "quiz": (
        "Generate {count} multiple-choice quiz questions for the topic: \"{topic}\".\n\n"
        "Output ONLY a JSON array. No commentary, no markdown fences. "
        "Each question has exactly these keys:\n"
        '  {{"q": "<question>", "choices": ["A", "B", "C", "D"], '
        '"correct": <index 0-3>, "explanation": "<why, with FAA citation>"}}\n\n'
        "Mix difficulty: a few recall, a few applied/scenario. Plausible distractors. "
        "Generate exactly {count} questions — no more, no fewer."
    ),
}

# Default counts and bounds.
_DEFAULT_COUNT = {"study": 1, "flashcards": 8, "quiz": 5}
_MAX_COUNT = {"study": 1, "flashcards": 50, "quiz": 50}


@app.route("/api/generate", methods=["POST"])
def generate():
    """Generate a study guide, flash cards, or quiz for one PPL topic."""
    if client is None:
        return jsonify(error="ANTHROPIC_API_KEY is not set."), 500

    data = request.get_json(silent=True) or {}
    kind = data.get("kind")
    topic = (data.get("topic") or "").strip()

    if kind not in _PROMPTS:
        return jsonify(error="`kind` must be 'study', 'flashcards', or 'quiz'."), 400
    if not topic:
        return jsonify(error="`topic` is required."), 400

    # Optional count for flashcards / quiz, clamped to a safe range.
    try:
        count = int(data.get("count") or _DEFAULT_COUNT[kind])
    except (TypeError, ValueError):
        count = _DEFAULT_COUNT[kind]
    count = max(1, min(count, _MAX_COUNT[kind]))

    # Pull RAG context for the topic.
    citations: list[str] = []
    context_block = ""
    if retrieval.index_ready():
        try:
            hits = retrieval.retrieve(topic, top_k=8)
            if hits:
                context_block = retrieval.format_context(hits, max_chars=10000)
                citations = [h.citation() for h in hits]
        except Exception:  # noqa: BLE001
            log.exception("Retrieval failed in /api/generate; continuing without context")

    # Shared tone guideline for all generated study material.
    tone_rules = (
        "Tone: semi-formal, professional instructor voice. Avoid colloquialisms "
        "and casual idioms (e.g., 'bite you', 'gonna', 'crush it', 'ace it', "
        "'sweat it', 'tricky bits', 'gotcha'). Write the way a CFI would write "
        "a printed study handout — clear, precise, professional."
    )

    system_prompt = (
        "You are an expert CFI creating accurate PPL study material for a student "
        "preparing for the checkride. Use the FAA excerpts below as your source of "
        "truth — do not invent regulations or page numbers.\n\n"
        f"{tone_rules}\n\n"
        f"FAA EXCERPTS:\n{context_block}"
        if context_block
        else
        "You are an expert CFI creating accurate PPL study material for a student. "
        "(No FAA excerpts available — use general aviation knowledge and flag any "
        "uncertain claims.)\n\n"
        f"{tone_rules}"
    )
    user_prompt = _PROMPTS[kind].format(topic=topic, count=count)

    # Bigger requests need more output room. Roughly 150 tokens per quiz/flash
    # item plus overhead.
    max_tokens = 4096 if kind == "study" else min(8192, 400 + count * 180)

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        )
    except APIError as e:
        log.exception("Claude API error in /api/generate")
        return jsonify(error=f"Claude API error: {e}"), 502
    except Exception as e:  # noqa: BLE001
        log.exception("Unexpected server error in /api/generate")
        return jsonify(error=f"Server error: {e}"), 500

    # Study guide is just markdown; the others are structured JSON.
    if kind == "study":
        return jsonify(kind=kind, topic=topic, content=text, citations=citations)

    cleaned = _strip_code_fence(text)
    try:
        items = _json.loads(cleaned)
    except _json.JSONDecodeError:
        return jsonify(
            error=f"Could not parse {kind} JSON from the model. Try again.",
            raw=text,
        ), 500

    return jsonify(kind=kind, topic=topic, items=items, citations=citations)


# --- /api/grade -------------------------------------------------------------
# Used by the DPE Oral exam to grade a student's answer against an "ideal"
# answer the topic curator wrote. Returns a verdict (correct / partial /
# incorrect), specific feedback, and a flag indicating whether a follow-up
# would be educational.

@app.route("/api/grade", methods=["POST"])
def grade():
    """Grade a DPE oral answer."""
    if client is None:
        return jsonify(error="ANTHROPIC_API_KEY is not set."), 500

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    answer = (data.get("answer") or "").strip()
    ideal = (data.get("ideal") or "").strip()  # optional now
    difficulty = (data.get("difficulty") or "checkride").strip()  # beginner/intermediate/checkride

    if not question or not answer:
        return jsonify(error="`question` and `answer` are required."), 400

    # Pull RAG context to ground the feedback in real FAA sources.
    context_block = ""
    if retrieval.index_ready():
        try:
            hits = retrieval.retrieve(question + " " + ideal, top_k=4)
            if hits:
                context_block = retrieval.format_context(hits, max_chars=4000)
        except Exception:  # noqa: BLE001
            log.exception("Retrieval failed in /api/grade")

    system_prompt = (
        f"You are a Designated Pilot Examiner (DPE) running a {difficulty}-level "
        f"practice oral with a student preparing for the Private Pilot checkride. "
        f"This is a conversational oral exam, not a written test. Grade the answer "
        f"and then continue the conversation the way a real DPE would.\n\n"

        f"=== GRADING PHILOSOPHY ===\n"
        f"- Mark 'correct' as long as the student covered the CORE of the question. "
        f"Be generous on details. If the gist is right, that's a pass.\n"
        f"- Mark 'incorrect' only when the core understanding is wrong, the student "
        f"contradicted a rule, or didn't actually address the question.\n"
        f"- A student admitting 'I don't know' is incorrect.\n\n"

        f"=== HOW THE DPE BEHAVES — STUDY THESE PATTERNS ===\n\n"

        f"** BEGINNER MODE **\n"
        f"Accept the core, then chain a RELATED follow-on question on the same broad "
        f"topic — not a drill-down on a missed detail, just the natural next thing in "
        f"the topic area. Brief affirmations: 'good', 'yes'.\n"
        f"Example chain:\n"
        f"  Q: 'What are the requirements to become a Private Pilot?'\n"
        f"  A: [list of requirements]\n"
        f"  DPE: 'Good — now on that topic, once you are rated, what must you do to "
        f"maintain currency?'\n"
        f"  A: 'Flight review every 24 months.'\n"
        f"  DPE: 'And what must you do to remain current to carry passengers?'\n"
        f"  A: '3 landings in 90 days.'\n"
        f"  DPE: 'Good.' [topic switches]\n"
        f"In Beginner, the next_question is a LATERAL move within the topic area, not "
        f"a drill into a missing detail. Keep it to one short, friendly sentence.\n\n"

        f"** INTERMEDIATE MODE **\n"
        f"Accept the core, then probe in three flavors (mix them):\n"
        f"  1. Detail probe: 'And must these landings be to a full stop?'\n"
        f"  2. Scenario variation: 'What if they're at night?'\n"
        f"  3. JUDGMENT test: 'Having just met these requirements, would you feel "
        f"comfortable flying passengers to a new airport?' — testing whether they "
        f"understand proficiency vs currency.\n"
        f"If the student gives a WRONG JUDGMENT (e.g. says yes to the comfort "
        f"question), firmly correct them and pivot to a teaching question: 'No, you "
        f"should not. Do you know the difference between proficiency and currency?' — "
        f"that becomes the next_question even though the answer was wrong.\n\n"

        f"** CHECKRIDE MODE (advanced/hard) **\n"
        f"Drill exhaustively. After each correct answer, push for completeness:\n"
        f"  - 'What else?' — when a list is incomplete\n"
        f"  - 'And what does that entail?' — drill into specifics\n"
        f"  - STACK multiple related sub-questions in one turn when natural:\n"
        f"    'Must these be to a full stop? And what about tail-wheel aircraft? "
        f"And what about night currency?'\n"
        f"Don't switch topics until the student has demonstrated comprehensive "
        f"knowledge of every aspect. Be rigorous but still respectful — you're an "
        f"examiner, not a bully.\n\n"

        f"=== TONE ===\n"
        f"- All difficulties: speak in a semi-formal, professional examiner tone. "
        f"Avoid colloquialisms and casual idioms like 'bite you', 'gonna', 'sweat it', "
        f"'no biggie', 'crush it', 'ace it', 'tricky bits', 'gotcha'. Use precise, "
        f"professional language a real DPE would use in an oral exam.\n"
        f"- beginner: encouraging but professional; brief, warm affirmations\n"
        f"- intermediate: balanced examiner — fair, conversational, corrective when needed\n"
        f"- checkride: rigorous DPE, demanding mastery, formal, stays respectful\n\n"

        + (f"FAA EXCERPTS for ground truth — use these for accuracy:\n{context_block}\n\n" if context_block else "")

        + "=== OUTPUT FORMAT ===\n"
        "Output ONLY a JSON object with these exact keys:\n"
        '  {"verdict": "correct" | "incorrect",\n'
        '   "score": <0-100>,\n'
        '   "feedback": "<conversational, second person. For correct answers, brief affirmation (1 sentence or less — \'good\', \'yes\', \'solid\'). For incorrect, 1-3 sentences correcting and teaching the key concept. Cite FAA sources where helpful but don\'t force it.>",\n'
        '   "next_question": "<the next question to ask, per the difficulty pattern above. For Beginner: a lateral follow-on within the topic area. For Intermediate: a detail probe, scenario variation, or judgment question. For Checkride: a drill-down or a multi-part stacked question. Can be null if the conversation has naturally exhausted the topic.>"}\n\n'
        "No commentary, no markdown fences."
    )
    user_prompt = (
        f"QUESTION: {question}\n\n"
        f"STUDENT'S ANSWER: {answer}"
        + (f"\n\nCURATOR'S REFERENCE ANSWER (for your context, not to be quoted): {ideal}" if ideal else "")
    )

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=600,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        )
    except APIError as e:
        log.exception("Claude API error in /api/grade")
        return jsonify(error=f"Claude API error: {e}"), 502
    except Exception as e:  # noqa: BLE001
        log.exception("Unexpected server error in /api/grade")
        return jsonify(error=f"Server error: {e}"), 500

    cleaned = _strip_code_fence(text)
    try:
        result = _json.loads(cleaned)
    except _json.JSONDecodeError:
        return jsonify(
            error="Could not parse grading JSON from the model.",
            raw=text,
        ), 500

    return jsonify(result)


# --- Entrypoint -------------------------------------------------------------
if __name__ == "__main__":
    # Render sets PORT; default to 8000 locally.
    port = int(os.environ.get("PORT", 8000))
    # debug=True is fine for local dev; gunicorn handles prod (see render.yaml).
    app.run(host="0.0.0.0", port=port, debug=True)
