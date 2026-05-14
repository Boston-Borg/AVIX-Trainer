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
        f"This is a conversational oral exam, not a written test. Grade the "
        f"student's answer, then decide whether a follow-up question is warranted.\n\n"

        f"=== THE THREE VERDICTS (READ CAREFULLY) ===\n"

        f"Be generous. The student is in a conversational oral, not a written exam. "
        f"Different phrasing, paraphrasing, and informal wording that means the "
        f"same thing as the reference answer all count as covered. Do not nitpick "
        f"on word choice. If you are torn between two verdicts, pick the more "
        f"generous one.\n\n"

        f"- 'correct'   = the student covered essentially all of the required "
        f"information for this difficulty and reached the right conclusion. "
        f"Minor phrasing differences are fine. Synonyms count. next_question "
        f"MUST be null. Feedback is a brief one-sentence affirmation ('Good.', "
        f"'Yes.', 'Solid.'). Do not lecture.\n"

        f"- 'partial'   = the student covered the BASE / CORE of the question "
        f"correctly AND roughly 70% or more of the required information for this "
        f"difficulty — but some specific required item(s) are missing. They are "
        f"NOT wrong; they are incomplete. next_question MUST be a focused "
        f"follow-up targeting the specific missing item(s). Feedback is empty "
        f"or null — the next_question alone is the response. CRITICALLY: do "
        f"NOT preface with acknowledgment ('Good, you covered the key elements "
        f"— one clarification...', 'You got most of it. However...', 'Let me "
        f"correct that.'). That pattern is banned. Simply ask the focused "
        f"follow-up that probes the gap.\n"

        f"- 'incorrect' = reserved for genuine failures. Use ONLY when one of "
        f"these is true:\n"
        f"     (a) the student stated something that contradicts an FAA rule "
        f"(e.g., 'VFR fuel reserve at night is 30 minutes'),\n"
        f"     (b) the student fundamentally misunderstood the question (e.g., "
        f"answering about IFR weather minimums when asked about VFR),\n"
        f"     (c) the student said 'I don't know', 'skip', or refused to "
        f"engage,\n"
        f"     (d) the student covered LESS than roughly 50% of the required "
        f"information AND the core/base of the question is also missing.\n"
        f"   For 'incorrect', feedback explains the correct concept in 1-3 "
        f"sentences. next_question should be null — the DPE moves on.\n\n"

        f"If you're uncertain whether to mark partial or incorrect, pick "
        f"partial. The base of the answer being there earns the student "
        f"partial credit and a follow-up, not a red mark.\n\n"

        f"=== KEY RULE FOR PARTIAL ===\n"
        f"When the verdict is 'partial', you have ONE behavior: put the focused "
        f"follow-up question into next_question, leave feedback empty. The "
        f"student sees ONLY the follow-up. You do NOT acknowledge what they "
        f"got right. You do NOT preview what they missed. You ask the question "
        f"that probes the gap. Period.\n\n"

        f"=== GENEROSITY SCALES WITH DIFFICULTY ===\n"
        f"- Beginner: maximum generosity. The base answer earns correct even "
        f"  if some named elements are skipped. Partial only when the student "
        f"  clearly missed something a private pilot must know.\n"
        f"- Intermediate: balanced. The base + most key conditions earns "
        f"  correct; partial when meaningful required details are missing.\n"
        f"- Checkride/Advanced: stricter but still partial-first. Correct "
        f"  requires comprehensive coverage including alternative paths and "
        f"  edge cases; anything short of that is partial.\n\n"

        f"=== CALIBRATION EXAMPLE (use this to anchor your scoring) ===\n"
        f"Question: 'You pull up this METAR. Tell me what you see and whether "
        f"you'd launch.'\n"
        f"METAR: 'KXYZ 121856Z 12015G25KT 3SM BR BKN008 OVC020 22/21 A2992'\n\n"
        f"Sample student answer: 'Airport ident, time, winds 121 at 15 gusting "
        f"to 25 kts, 3sm visibility, skies broken at 800 feet, overcast 2k, "
        f"temp 22 dewpoint 21, altimeter 2992. I would not fly as these are "
        f"IFR conditions with very high winds and mist.'\n\n"
        f"How this should be graded:\n"
        f"  • At BEGINNER: verdict = 'correct'. The student decoded every "
        f"    field and made the right go/no-go call. Feedback: 'Good.' "
        f"    next_question: null.\n"
        f"  • At INTERMEDIATE: verdict = 'partial'. The base is right; "
        f"    missing is explicit reasoning about the 1°C temp/dew-point "
        f"    spread and the fog risk that creates. next_question = a "
        f"    targeted follow-up like 'What's that 1-degree spread telling "
        f"    you about the next hour?'. Feedback: empty.\n"
        f"  • At CHECKRIDE: verdict = 'partial'. Missing is the spread "
        f"    analysis, gust factor consideration, and alternate planning. "
        f"    next_question = a stacked follow-up like 'With that 1° spread, "
        f"    what would you expect in the next hour, and how does that "
        f"    affect your alternate selection?'. Feedback: empty.\n\n"
        f"This answer must NEVER be graded 'incorrect'. The student is "
        f"substantively right.\n\n"

        f"=== HOW MUCH IS 'REQUIRED' DEPENDS ON DIFFICULTY ===\n"
        f"The bar for 'required information' rises with difficulty. The same "
        f"question produces different follow-up behavior at different levels.\n\n"

        f"** BEGINNER **\n"
        f"Required = the headline / big-picture answer only. Major rule items, "
        f"not specific edge cases.\n"
        f"Example — Q: 'What are the currency requirements to carry passengers?'\n"
        f"  Required at Beginner: (1) flight review every 24 calendar months, "
        f"AND (2) 3 takeoffs and landings in the preceding 90 days.\n"
        f"  If the student mentioned BOTH → next_question: null.\n"
        f"  If they mentioned only one → ask about the missing one in one short, "
        f"friendly sentence.\n\n"

        f"** INTERMEDIATE **\n"
        f"Required = all the Beginner-level items PLUS the more specific "
        f"variations and conditions that apply in everyday flying.\n"
        f"Example — same currency question:\n"
        f"  Required at Intermediate: Beginner items + the night-currency rule "
        f"(3 takeoffs/landings to a FULL STOP, between 1 hr after sunset and "
        f"1 hr before sunrise) + the tailwheel rule (all landings to a full stop).\n"
        f"  If any of these are missing → ask a focused probe targeting the gap.\n"
        f"  If all are covered → next_question: null.\n\n"

        f"** ADVANCED / CHECKRIDE **\n"
        f"Required = all the Intermediate-level items PLUS deeper specifics, "
        f"alternative compliance paths, and edge cases.\n"
        f"Example — same currency question:\n"
        f"  Required at Advanced: Intermediate items + the FAA Wings program as an "
        f"alternative path (and what it entails — phased flights with a CFI plus "
        f"ground lessons), 61.57 high-altitude considerations, type-rating and "
        f"category/class nuances where relevant.\n"
        f"  If multiple items are missing, you may stack 2-3 sub-questions in a "
        f"single follow-up turn: e.g., 'What about night currency? And tailwheel? "
        f"And what other paths can satisfy the flight review requirement?'\n"
        f"  If everything is covered → next_question: null.\n\n"

        f"=== STYLE WHEN A FOLLOW-UP IS WARRANTED ===\n"
        f"- Beginner: one short, friendly sentence targeting the missing item.\n"
        f"- Intermediate: one focused probe, can include a scenario condition "
        f"(e.g., 'And what about at night?').\n"
        f"- Advanced: a focused probe, or multi-part if several required items "
        f"are missing.\n\n"

        f"=== TONE ===\n"
        f"- All difficulties: semi-formal, professional examiner. Avoid "
        f"colloquialisms and casual idioms ('bite you', 'gonna', 'sweat it', "
        f"'crush it', 'gotcha', 'tricky bits', etc.). Use precise, professional "
        f"language a real DPE would use in an oral exam.\n"
        f"- beginner: encouraging but professional; brief, warm affirmations\n"
        f"- intermediate: balanced examiner — fair, corrective when needed\n"
        f"- checkride: rigorous DPE, demanding mastery, formal, respectful\n\n"

        + (f"FAA EXCERPTS for ground truth — use these for accuracy:\n{context_block}\n\n" if context_block else "")

        + "=== OUTPUT FORMAT ===\n"
        "Output ONLY a JSON object with these exact keys:\n"
        '  {"verdict": "correct" | "partial" | "incorrect",\n'
        '   "score": <0-100; correct ~90-100, partial ~50-80, incorrect ~0-40>,\n'
        '   "feedback": "<For correct: brief affirmation (1 sentence). For partial: empty string or null — the next_question itself is the response, do NOT acknowledge partial correctness. For incorrect: 1-3 sentences explaining the correct concept.>",\n'
        '   "next_question": "<REQUIRED when verdict=\'partial\' — the focused follow-up that probes the specific missing required information. MUST be null when verdict=\'correct\' or verdict=\'incorrect\'. Never invent lateral, judgment, or scenario questions just to continue the conversation.>"}\n\n'
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
