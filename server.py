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


# --- Supabase client setup --------------------------------------------------
# Supabase handles user accounts and per-user data. The publishable (anon) key
# is safe to use here — Row Level Security on the Supabase side controls what
# each user can actually read/write. Same "build only if configured" pattern
# as Anthropic above, so the server still boots if you forget to set the keys.
from supabase import create_client
from functools import wraps
from datetime import datetime

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
# Service role key — used ONLY by webhook handlers that need to write rows
# on behalf of the user (RLS would otherwise block them). Never exposed
# to the browser. Keep in .env / Render's encrypted env store.
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

supabase = (
    create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    if (SUPABASE_URL and SUPABASE_ANON_KEY)
    else None
)
# Admin client — used by Stripe webhook handlers to write subscription rows.
supabase_admin = (
    create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    if (SUPABASE_URL and SUPABASE_SERVICE_KEY)
    else None
)


# --- Stripe client setup ---------------------------------------------------
# Stripe handles all card data — we never touch raw card numbers. The
# checkout flow is hosted by Stripe (PCI-compliant for us by default).
import stripe

STRIPE_SECRET_KEY     = os.environ.get("STRIPE_SECRET_KEY")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY")
STRIPE_PRICE_ID       = os.environ.get("STRIPE_PRICE_ID")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")
STRIPE_SUCCESS_URL = os.environ.get(
    "STRIPE_SUCCESS_URL",
    "http://localhost:8000/?checkout=success&session_id={CHECKOUT_SESSION_ID}",
)
STRIPE_CANCEL_URL = os.environ.get(
    "STRIPE_CANCEL_URL", "http://localhost:8000/?checkout=cancel"
)

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY


def _stripe_configured() -> bool:
    """True when we have enough Stripe config to actually charge for things."""
    return bool(STRIPE_SECRET_KEY and STRIPE_PRICE_ID)

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
        supabase_configured=supabase is not None,
        supabase_admin_configured=supabase_admin is not None,
        stripe_configured=_stripe_configured(),
        stripe_webhook_configured=bool(STRIPE_WEBHOOK_SECRET),
        model=CLAUDE_MODEL,
        rag_index_loaded=retrieval.index_ready(),
        rag_chunks=len(retrieval._index.chunks),
    )


# --- Auth: signup, login, logout, "who am I" --------------------------------
# The flow:
#   1. Browser POSTs to /api/signup or /api/login with {email, password}.
#   2. We hand the credentials to Supabase. If valid, Supabase returns an
#      access token (a JWT). We pass that token back to the browser.
#   3. The browser stores the token (in localStorage) and includes it in the
#      "Authorization: Bearer <token>" header on every later API call.
#   4. The @require_auth decorator below verifies that header on protected
#      routes by asking Supabase "is this token still valid, and who does it
#      belong to?" — and attaches the user info to the Flask request object
#      so route handlers can read it as `request.user`.


def require_auth(f):
    """Decorator: rejects the request with 401 if no valid Supabase JWT.

    Usage:
        @app.route("/api/chat", methods=["POST"])
        @require_auth
        def chat():
            user = request.user  # populated by this decorator
            ...
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        if supabase is None:
            return (
                jsonify(error="Auth not configured: SUPABASE_URL / SUPABASE_ANON_KEY missing."),
                500,
            )
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify(error="Missing or malformed Authorization header."), 401
        token = auth_header.split(" ", 1)[1].strip()
        if not token:
            return jsonify(error="Empty bearer token."), 401
        try:
            user_response = supabase.auth.get_user(token)
        except Exception:  # noqa: BLE001
            log.exception("Token verification failed")
            return jsonify(error="Invalid or expired token."), 401
        if user_response is None or user_response.user is None:
            return jsonify(error="Invalid or expired token."), 401

        # Attach the user (and the raw JWT) to the request so route handlers
        # can use them. request.user.id is what you'd save as a row owner.
        request.user = user_response.user
        request.access_token = token
        return f(*args, **kwargs)

    return wrapper


def _user_payload(user):
    """Shrink a Supabase user object down to the safe fields we send to the
    browser. Don't ever send the whole `user` — it can include internal flags."""
    return {"id": user.id, "email": user.email}


@app.route("/api/signup", methods=["POST"])
def signup():
    """Create a new user account.

    Body: {"email": "...", "password": "..."}
    Returns: {"access_token": "...", "user": {...}}
        or  {"message": "Check your email to confirm your account."} if email
        confirmation is enabled in Supabase (default for new projects).
    """
    if supabase is None:
        return jsonify(error="Auth not configured on server."), 500
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify(error="Email and password are required."), 400
    if len(password) < 8:
        return jsonify(error="Password must be at least 8 characters."), 400

    try:
        result = supabase.auth.sign_up({"email": email, "password": password})
    except Exception as e:  # noqa: BLE001
        log.exception("Signup failed")
        # Supabase exception messages are user-friendly ("User already
        # registered", "Password should be at least..."), safe to surface.
        return jsonify(error=str(e)), 400

    if result.session is None:
        # Email confirmation is on; user must click the link before login.
        return jsonify(
            message="Account created. Check your email to confirm before logging in."
        )

    return jsonify(
        access_token=result.session.access_token,
        user=_user_payload(result.user),
    )


@app.route("/api/login", methods=["POST"])
def login():
    """Sign in with email + password. Returns a JWT for the browser to keep."""
    if supabase is None:
        return jsonify(error="Auth not configured on server."), 500
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify(error="Email and password are required."), 400

    try:
        result = supabase.auth.sign_in_with_password(
            {"email": email, "password": password}
        )
    except Exception:  # noqa: BLE001
        # Don't leak whether email exists vs wrong password — give the same
        # generic message either way.
        log.info("Login failed for email=%r", email)
        return jsonify(error="Invalid email or password."), 401

    return jsonify(
        access_token=result.session.access_token,
        user=_user_payload(result.user),
    )


@app.route("/api/logout", methods=["POST"])
@require_auth
def logout():
    """Sign out — revokes the user's session in Supabase.

    The browser should also delete its local copy of the token after a 200.
    """
    try:
        supabase.auth.sign_out()
    except Exception:  # noqa: BLE001
        # Worst case the JWT keeps working until its natural expiry; that's
        # OK because the browser will throw away its copy regardless.
        log.exception("Server-side sign_out failed; client-side discard still OK")
    return jsonify(message="Logged out.")


@app.route("/api/me", methods=["GET"])
@require_auth
def me():
    """Return the currently-logged-in user + subscription status.
    The frontend uses this on page load to decide between three screens:
      - logged-out  → show login
      - logged-in, no sub → show paywall
      - logged-in, active sub → show app"""
    sub_status = (
        get_subscription_status(request.user.id)
        if _stripe_configured()
        else {"active": True, "status": "stripe_not_configured"}
    )
    return jsonify(
        user=_user_payload(request.user),
        subscription=sub_status,
        stripe_publishable_key=STRIPE_PUBLISHABLE_KEY or "",
    )


# --- Subscriptions ---------------------------------------------------------
# Status lookup + paywall decorator. When Stripe is unconfigured the gate is
# disabled (so local dev works before you set up your Stripe account).

def get_subscription_status(user_id: str) -> dict:
    """Read the user's subscription row from Supabase. Returns a small dict
    summarizing whether they have access."""
    client = supabase_admin or supabase
    if client is None:
        return {"active": False, "status": "supabase_not_configured"}
    try:
        result = (
            client.table("subscriptions")
            .select("status, current_period_end, cancel_at_period_end")
            .eq("user_id", str(user_id))
            .limit(1)
            .execute()
        )
        rows = result.data or []
    except Exception:  # noqa: BLE001
        log.exception("Failed to read subscriptions row")
        return {"active": False, "status": "lookup_error"}

    if not rows:
        return {"active": False, "status": "none"}
    row = rows[0]
    active_states = {"active", "trialing"}
    return {
        "active": row.get("status") in active_states,
        "status": row.get("status"),
        "current_period_end": row.get("current_period_end"),
        "cancel_at_period_end": bool(row.get("cancel_at_period_end")),
    }


def require_subscription(f):
    """Block the route with HTTP 402 if the user has no active subscription.

    Must be stacked below @require_auth (so request.user is populated).
    No-ops when Stripe isn't configured yet, so local dev keeps working
    until you finish your Stripe setup."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not _stripe_configured():
            return f(*args, **kwargs)  # paywall off until Stripe is wired up
        user = getattr(request, "user", None)
        if user is None:
            return jsonify(error="Unauthenticated."), 401
        status = get_subscription_status(user.id)
        if not status.get("active"):
            return (
                jsonify(error="Active subscription required.", subscription=status),
                402,  # Payment Required
            )
        return f(*args, **kwargs)

    return wrapper


# --- Stripe endpoints ------------------------------------------------------
# 1. /api/stripe/create-checkout-session — frontend hits this when user clicks
#    "Subscribe". We create a hosted Stripe Checkout session and return its
#    URL; the browser redirects there. Stripe collects the card, charges it,
#    and redirects the user back to STRIPE_SUCCESS_URL on completion.
# 2. /api/stripe/sync-after-checkout — frontend hits this on the success
#    redirect. Looks up the checkout session on Stripe, verifies the user
#    actually paid, mirrors the subscription state into our DB. Works even
#    without webhooks configured — useful for local dev.
# 3. /api/stripe/webhook — Stripe POSTs events here for the full lifecycle
#    (renewals, cancellations, failed payments). Source of truth in prod.
# 4. /api/stripe/portal — opens Stripe's hosted Customer Portal so users can
#    update payment method, cancel, view invoices, etc.

def _upsert_subscription_from_stripe(user_id: str, sub) -> None:
    """Write/refresh a row in public.subscriptions from a Stripe subscription object."""
    if supabase_admin is None:
        log.warning(
            "Cannot persist subscription: SUPABASE_SERVICE_KEY not set. "
            "Add it to .env (Supabase Project Settings → API Keys → service_role)."
        )
        return
    # Stripe sometimes returns ids vs full objects depending on how the call
    # was made. If we got just an id back, fetch the full object.
    if isinstance(sub, str):
        try:
            sub = stripe.Subscription.retrieve(sub)
        except Exception:  # noqa: BLE001
            log.exception("Could not retrieve Stripe subscription id=%s", sub)
            return

    # Stripe-py v10+ objects don't support .get() — use [] / KeyError instead.
    # This helper handles both Stripe objects and plain dicts.
    def _sf(obj, key, default=None):
        try:
            return obj[key]
        except (KeyError, TypeError):
            return default

    period_end_ts = _sf(sub, "current_period_end")
    if not period_end_ts:
        # current_period_end moved to items.data[0] in newer Stripe API versions.
        try:
            period_end_ts = sub["items"]["data"][0]["current_period_end"]
        except (KeyError, IndexError, TypeError):
            period_end_ts = None

    period_end_iso = (
        datetime.utcfromtimestamp(period_end_ts).isoformat() + "Z"
        if period_end_ts
        else None
    )
    payload = {
        "user_id": str(user_id),
        "stripe_customer_id": _sf(sub, "customer"),
        "stripe_subscription_id": _sf(sub, "id"),
        "status": _sf(sub, "status"),
        "current_period_end": period_end_iso,
        "cancel_at_period_end": bool(_sf(sub, "cancel_at_period_end", False)),
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    log.info(
        "Upserting subscription for user=%s status=%s sub_id=%s customer=%s",
        payload["user_id"], payload["status"], payload["stripe_subscription_id"],
        payload["stripe_customer_id"],
    )
    try:
        result = supabase_admin.table("subscriptions").upsert(
            payload, on_conflict="user_id"
        ).execute()
        log.info("Upsert result: %r", getattr(result, "data", None))
    except Exception as e:  # noqa: BLE001
        log.exception(
            "Failed to upsert subscriptions row. payload=%r error=%s", payload, e
        )


@app.route("/api/stripe/create-checkout-session", methods=["POST"])
@require_auth
def stripe_create_checkout():
    if not _stripe_configured():
        return jsonify(error="Stripe is not configured on the server."), 500
    user = request.user
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer_email=user.email,
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            success_url=STRIPE_SUCCESS_URL,
            cancel_url=STRIPE_CANCEL_URL,
            client_reference_id=user.id,
            # Metadata travels with the session AND the resulting subscription
            # so our webhook can map Stripe events back to the Supabase user.
            metadata={"supabase_user_id": user.id},
            subscription_data={"metadata": {"supabase_user_id": user.id}},
            allow_promotion_codes=True,
        )
        return jsonify(url=session.url)
    except Exception as e:  # noqa: BLE001
        log.exception("Stripe checkout session creation failed")
        return jsonify(error=str(e)), 400


@app.route("/api/stripe/sync-after-checkout", methods=["POST"])
@require_auth
def stripe_sync_after_checkout():
    """Frontend calls this on the success redirect. Verifies the session
    actually belongs to the current user and was paid, then mirrors the
    subscription into our DB. Lets us flip the paywall off immediately
    without needing webhooks set up."""
    if not _stripe_configured():
        return jsonify(error="Stripe is not configured."), 500
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    if not session_id:
        return jsonify(error="session_id is required"), 400
    try:
        session = stripe.checkout.Session.retrieve(
            session_id, expand=["subscription"]
        )
    except Exception as e:  # noqa: BLE001
        log.exception("Stripe session retrieval failed")
        return jsonify(error=str(e)), 400

    meta_user = (session.get("metadata") or {}).get("supabase_user_id")
    if meta_user and meta_user != request.user.id:
        return jsonify(error="Checkout session does not belong to this user."), 403
    # "paid" = card charged successfully.
    # "no_payment_required" = $0 total (e.g. 100% promo code) — still a valid completion.
    if session.get("payment_status") not in ("paid", "no_payment_required"):
        return jsonify(error="Payment is not complete yet."), 400

    sub = session.get("subscription")
    if sub and not isinstance(sub, str):
        _upsert_subscription_from_stripe(request.user.id, sub)
    return jsonify(
        synced=True, subscription=get_subscription_status(request.user.id)
    )


@app.route("/api/stripe/webhook", methods=["POST"])
def stripe_webhook():
    """Stripe POSTs subscription lifecycle events here. We verify the signature
    (proves it's really from Stripe) and update our subscriptions table."""
    if not STRIPE_WEBHOOK_SECRET:
        # The endpoint exists but is inert until you configure the secret.
        # This means locally you can test the happy path via sync-after-checkout
        # before bothering with stripe-cli.
        return jsonify(error="Webhook secret not configured"), 503
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", "")
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except (ValueError, stripe.error.SignatureVerificationError):
        log.warning("Stripe webhook signature verification failed")
        return jsonify(error="Invalid signature"), 400

    event_type = event["type"]
    obj = event["data"]["object"]
    log.info("Stripe webhook: %s", event_type)

    if event_type == "checkout.session.completed":
        user_id = (obj.get("metadata") or {}).get("supabase_user_id")
        sub_id = obj.get("subscription")
        if user_id and sub_id:
            try:
                sub = stripe.Subscription.retrieve(sub_id)
                _upsert_subscription_from_stripe(user_id, sub)
            except Exception:  # noqa: BLE001
                log.exception("Failed to sync from checkout.session.completed")

    elif event_type in (
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
    ):
        user_id = (obj.get("metadata") or {}).get("supabase_user_id")
        # If metadata is missing (e.g. older subscriptions), fall back to
        # mapping the customer ID we previously saved.
        if not user_id and supabase_admin is not None:
            customer_id = obj.get("customer")
            if customer_id:
                try:
                    res = (
                        supabase_admin.table("subscriptions")
                        .select("user_id")
                        .eq("stripe_customer_id", customer_id)
                        .limit(1)
                        .execute()
                    )
                    if res.data:
                        user_id = res.data[0]["user_id"]
                except Exception:  # noqa: BLE001
                    log.exception("Customer-id lookup failed")
        if user_id:
            _upsert_subscription_from_stripe(user_id, obj)

    return jsonify(received=True)


@app.route("/api/stripe/portal", methods=["POST"])
@require_auth
def stripe_portal():
    """Open Stripe's hosted Customer Portal for managing the subscription."""
    if not _stripe_configured():
        return jsonify(error="Stripe is not configured."), 500
    client = supabase_admin or supabase
    if client is None:
        return jsonify(error="Supabase is not configured."), 500
    try:
        res = (
            client.table("subscriptions")
            .select("stripe_customer_id")
            .eq("user_id", str(request.user.id))
            .limit(1)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        log.exception("Subscription lookup for portal failed")
        return jsonify(error=str(e)), 500

    if not res.data or not res.data[0].get("stripe_customer_id"):
        return jsonify(error="No subscription found for this account."), 404
    customer_id = res.data[0]["stripe_customer_id"]
    try:
        portal_session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=request.host_url,
        )
        return jsonify(url=portal_session.url)
    except Exception as e:  # noqa: BLE001
        log.exception("Stripe portal session creation failed")
        return jsonify(error=str(e)), 400


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
@require_auth
@require_subscription
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
@require_auth
@require_subscription
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

    system_prompt = (
        "You are an expert CFI creating accurate PPL study material for a student "
        "preparing for the checkride. Use the FAA excerpts below as your source of "
        "truth — do not invent regulations or page numbers.\n\n"
        f"FAA EXCERPTS:\n{context_block}"
        if context_block
        else
        "You are an expert CFI creating accurate PPL study material for a student. "
        "(No FAA excerpts available — use general aviation knowledge and flag any "
        "uncertain claims.)"
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
@require_auth
@require_subscription
def grade():
    """Grade a DPE oral answer."""
    if client is None:
        return jsonify(error="ANTHROPIC_API_KEY is not set."), 500

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    answer = (data.get("answer") or "").strip()
    ideal = (data.get("ideal") or "").strip()
    difficulty = (data.get("difficulty") or "checkride").strip()  # beginner/intermediate/checkride

    if not question or not answer or not ideal:
        return jsonify(error="`question`, `answer`, and `ideal` are required."), 400

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
        f"You are a Designated Pilot Examiner (DPE) grading a student's verbal "
        f"answer in a {difficulty}-level practice oral. You have the question, "
        f"the student's answer, and the curator's ideal answer.\n\n"
        f"Tone:\n"
        f"- beginner: encouraging coach, generous on partial credit\n"
        f"- intermediate: balanced examiner, fair but precise\n"
        f"- checkride: rigorous DPE, holds the line on safety-critical items\n\n"
        + (f"FAA EXCERPTS for ground truth:\n{context_block}\n\n" if context_block else "")
        + "Output ONLY a JSON object with these exact keys:\n"
        '  {"verdict": "correct" | "partial" | "incorrect",\n'
        '   "score": <0-100>,\n'
        '   "feedback": "<2-4 sentences, address the student in second person, cite FAA sources where helpful>",\n'
        '   "ask_followup": <true if a probing follow-up would be educational, false if the topic is fully covered>}\n\n'
        "No commentary, no markdown fences."
    )
    user_prompt = (
        f"QUESTION: {question}\n\n"
        f"STUDENT'S ANSWER: {answer}\n\n"
        f"IDEAL ANSWER (curator's reference, not to be quoted verbatim): {ideal}"
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
