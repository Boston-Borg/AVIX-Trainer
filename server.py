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


# --- Owner allowlist -------------------------------------------------------
# Emails listed here get unlimited free access (paywall bypassed). Used for
# project owners / co-founders. Comma-separated in the env var. Whitespace
# is trimmed and case is ignored at comparison time.
OWNER_EMAILS = {
    email.strip().lower()
    for email in (os.environ.get("OWNER_EMAILS") or "").split(",")
    if email.strip()
}


def is_owner(email) -> bool:
    """True if the user's email is in OWNER_EMAILS (case-insensitive)."""
    if not email:
        return False
    return str(email).strip().lower() in OWNER_EMAILS


# --- Free trial limits -----------------------------------------------------
# Non-subscribers get a one-shot taste of each feature so they can see what
# they'd be paying for. State lives in the trial_usage table.
TRIAL_CHAT_WINDOW_MINUTES = 5
TRIAL_ORAL_WINDOW_MINUTES = 30
TRIAL_GENERATE_LIMIT       = 1


def _parse_iso(s):
    """Parse a Supabase timestamptz string ('2026-05-15T...+00:00' or '...Z')."""
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00") if s.endswith("Z") else s
        return datetime.fromisoformat(s)
    except Exception:  # noqa: BLE001
        return None


def _read_trial_row(user_id):
    """Fetch the trial_usage row for a user. Returns dict or None."""
    if supabase_admin is None:
        return None
    try:
        res = (
            supabase_admin.table("trial_usage")
            .select("*")
            .eq("user_id", str(user_id))
            .limit(1)
            .execute()
        )
        return (res.data or [None])[0]
    except Exception:  # noqa: BLE001
        log.exception("Failed to read trial_usage row")
        return None


def _upsert_trial(user_id, **fields):
    """Upsert the user's trial_usage row with the given fields."""
    if supabase_admin is None:
        return
    payload = {
        "user_id": str(user_id),
        "updated_at": datetime.utcnow().isoformat() + "Z",
        **fields,
    }
    try:
        supabase_admin.table("trial_usage").upsert(
            payload, on_conflict="user_id"
        ).execute()
    except Exception:  # noqa: BLE001
        log.exception("Failed to upsert trial_usage row")


def get_trial_status(user_id) -> dict:
    """Snapshot of the user's trial state — used by /api/me so the frontend
    can show "5 minutes left" / "1 quiz left" etc."""
    row = _read_trial_row(user_id) or {}
    now = datetime.utcnow().replace(tzinfo=None)

    # Chat: window starts when they sent their first message.
    chat_first = _parse_iso(row.get("chat_first_at"))
    if chat_first is None:
        chat = {"used": False, "seconds_remaining": TRIAL_CHAT_WINDOW_MINUTES * 60}
    else:
        elapsed = (now - chat_first.replace(tzinfo=None)).total_seconds()
        remaining = max(0, TRIAL_CHAT_WINDOW_MINUTES * 60 - elapsed)
        chat = {
            "used": True,
            "seconds_remaining": int(remaining),
            "expired": remaining <= 0,
        }

    # Generate (study guide/flashcards/quiz): count-based.
    generate_count = int(row.get("generate_count") or 0)
    generate = {
        "used": generate_count,
        "remaining": max(0, TRIAL_GENERATE_LIMIT - generate_count),
        "expired": generate_count >= TRIAL_GENERATE_LIMIT,
    }

    # Oral exam: window from first graded answer.
    oral_first = _parse_iso(row.get("oral_first_at"))
    if oral_first is None:
        oral = {"used": False, "seconds_remaining": TRIAL_ORAL_WINDOW_MINUTES * 60}
    else:
        elapsed = (now - oral_first.replace(tzinfo=None)).total_seconds()
        remaining = max(0, TRIAL_ORAL_WINDOW_MINUTES * 60 - elapsed)
        oral = {
            "used": True,
            "seconds_remaining": int(remaining),
            "expired": remaining <= 0,
        }

    any_available = (
        not chat.get("expired", False)
        or not generate.get("expired", False)
        or not oral.get("expired", False)
    )
    return {
        "chat": chat,
        "generate": generate,
        "oral": oral,
        "any_available": any_available,
    }


def _check_trial_usage(user_id, feature: str) -> bool:
    """Check if the user can use `feature` under their trial allowance, and
    record usage if so. Returns True (allow) or False (deny / paywall)."""
    if supabase_admin is None:
        # Without admin client we can't track trial state. Fail closed so we
        # never give unintended free access in production.
        return False

    row = _read_trial_row(user_id) or {}
    now = datetime.utcnow()
    now_iso = now.isoformat() + "Z"

    if feature == "chat":
        first = _parse_iso(row.get("chat_first_at"))
        if first is None:
            _upsert_trial(user_id, chat_first_at=now_iso)
            return True
        elapsed = (now - first.replace(tzinfo=None)).total_seconds()
        return elapsed < TRIAL_CHAT_WINDOW_MINUTES * 60

    if feature == "generate":
        count = int(row.get("generate_count") or 0)
        if count < TRIAL_GENERATE_LIMIT:
            _upsert_trial(user_id, generate_count=count + 1)
            return True
        return False

    if feature == "grade":
        first = _parse_iso(row.get("oral_first_at"))
        if first is None:
            _upsert_trial(user_id, oral_first_at=now_iso)
            return True
        elapsed = (now - first.replace(tzinfo=None)).total_seconds()
        return elapsed < TRIAL_ORAL_WINDOW_MINUTES * 60

    # Unknown feature — deny.
    return False

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
      - logged-in, active sub → show app
      - logged-in, owner → show app (paywall bypassed)"""
    user = request.user
    if is_owner(user.email):
        sub_status = {"active": True, "status": "owner"}
        trial_status = None  # owners don't need a trial
    elif _stripe_configured():
        sub_status = get_subscription_status(user.id)
        # Only fetch trial status if the user isn't a paid subscriber.
        trial_status = (
            None if sub_status.get("active") else get_trial_status(user.id)
        )
    else:
        sub_status = {"active": True, "status": "stripe_not_configured"}
        trial_status = None
    return jsonify(
        user=_user_payload(user),
        subscription=sub_status,
        trial=trial_status,
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
        # Owners (founders / co-founders listed in OWNER_EMAILS) bypass paywall.
        if is_owner(user.email):
            return f(*args, **kwargs)
        status = get_subscription_status(user.id)
        if not status.get("active"):
            return (
                jsonify(error="Active subscription required.", subscription=status),
                402,  # Payment Required
            )
        return f(*args, **kwargs)

    return wrapper


def require_paid_access(feature: str):
    """Decorator factory: gate the route by owner-bypass, active subscription,
    OR a remaining trial allowance for the named feature.

    Args:
        feature: one of 'chat', 'generate', 'grade'.

    Must be stacked below @require_auth.

    Order of checks:
        1. Stripe not configured → allow (dev mode).
        2. Owner email           → allow.
        3. Active subscription   → allow.
        4. Trial allowance       → allow + record usage.
        5. Otherwise             → 402 Payment Required.
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not _stripe_configured():
                return f(*args, **kwargs)
            user = getattr(request, "user", None)
            if user is None:
                return jsonify(error="Unauthenticated."), 401
            if is_owner(user.email):
                return f(*args, **kwargs)
            sub = get_subscription_status(user.id)
            if sub.get("active"):
                return f(*args, **kwargs)
            if _check_trial_usage(user.id, feature):
                return f(*args, **kwargs)
            return (
                jsonify(
                    error=(
                        "Your free trial for this feature is used up. "
                        "Subscribe to continue."
                    ),
                    feature=feature,
                    subscription=sub,
                    trial=get_trial_status(user.id),
                ),
                402,  # Payment Required
            )

        return wrapper

    return decorator


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

def _stripe_get(obj, key, default=None):
    """Safe field access for Stripe-py v10+ objects (which removed .get()) and
    for plain dicts. Use this anywhere we touch a Stripe API response."""
    try:
        return obj[key]
    except (KeyError, TypeError):
        return default


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

    period_end_ts = _stripe_get(sub, "current_period_end")
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
        "stripe_customer_id": _stripe_get(sub, "customer"),
        "stripe_subscription_id": _stripe_get(sub, "id"),
        "status": _stripe_get(sub, "status"),
        "current_period_end": period_end_iso,
        "cancel_at_period_end": bool(_stripe_get(sub, "cancel_at_period_end", False)),
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

    metadata = _stripe_get(session, "metadata") or {}
    meta_user = _stripe_get(metadata, "supabase_user_id")
    if meta_user and meta_user != request.user.id:
        return jsonify(error="Checkout session does not belong to this user."), 403
    # "paid" = card charged successfully.
    # "no_payment_required" = $0 total (e.g. 100% promo code) — still a valid completion.
    payment_status = _stripe_get(session, "payment_status")
    if payment_status not in ("paid", "no_payment_required"):
        return jsonify(error=f"Payment is not complete yet (status={payment_status})."), 400

    sub = _stripe_get(session, "subscription")
    if sub and not isinstance(sub, str):
        _upsert_subscription_from_stripe(request.user.id, sub)
    elif isinstance(sub, str):
        # If the expand=["subscription"] didn't expand for any reason, fetch it.
        try:
            sub_obj = stripe.Subscription.retrieve(sub)
            _upsert_subscription_from_stripe(request.user.id, sub_obj)
        except Exception:  # noqa: BLE001
            log.exception("Could not retrieve subscription from id-only response")
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
        metadata = _stripe_get(obj, "metadata") or {}
        user_id = _stripe_get(metadata, "supabase_user_id")
        sub_id = _stripe_get(obj, "subscription")
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
        metadata = _stripe_get(obj, "metadata") or {}
        user_id = _stripe_get(metadata, "supabase_user_id")
        # If metadata is missing (e.g. older subscriptions), fall back to
        # mapping the customer ID we previously saved.
        if not user_id and supabase_admin is not None:
            customer_id = _stripe_get(obj, "customer")
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
@require_paid_access("chat")
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
@require_paid_access("generate")
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
@require_auth
@require_paid_access("grade")
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
