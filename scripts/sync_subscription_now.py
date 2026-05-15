"""
Recovery / diagnostic: take your most recent ACTIVE Stripe subscription and
mirror it into Supabase's subscriptions table, so the AVX paywall unlocks.

Use this when the sync-after-checkout endpoint silently failed and you need
to unblock yourself without going through another checkout.

Usage:
    python scripts/sync_subscription_now.py <email>

Example:
    python scripts/sync_subscription_now.py boston.borg@outlook.com
"""

import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dotenv import load_dotenv
load_dotenv()

import stripe
from supabase import create_client


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/sync_subscription_now.py <email>")
        sys.exit(1)
    email = sys.argv[1].strip().lower()

    # ---- Sanity-check the env ----
    stripe_key = os.environ.get("STRIPE_SECRET_KEY")
    supa_url   = os.environ.get("SUPABASE_URL")
    supa_svc   = os.environ.get("SUPABASE_SERVICE_KEY")

    if not stripe_key:
        print("ERROR: STRIPE_SECRET_KEY missing from .env"); sys.exit(1)
    if not supa_url or not supa_svc:
        print("ERROR: SUPABASE_URL or SUPABASE_SERVICE_KEY missing from .env")
        sys.exit(1)

    print(f"SUPABASE_SERVICE_KEY prefix: {supa_svc[:18]}...")
    if supa_svc.startswith("sb_publishable_"):
        print("⚠️  WARNING: that prefix looks like the PUBLISHABLE key, not service_role.")
        print("    You probably need to copy the `secret` / `service_role` key instead.")
        print("    (Supabase dashboard → Project Settings → API Keys → row labeled 'secret')")
        sys.exit(1)

    stripe.api_key = stripe_key
    supa = create_client(supa_url, supa_svc)

    # ---- Find the user in Supabase auth ----
    print(f"\nLooking up Supabase user by email: {email} ...")
    try:
        # Supabase admin API: list users, filter by email
        # The admin auth API exposes list_users() on supabase.auth.admin
        users_page = supa.auth.admin.list_users()
        # depending on supabase-py version, this returns a list of users
        # or a paginated object
        users = users_page if isinstance(users_page, list) else getattr(users_page, "users", [])
        match = next((u for u in users if (getattr(u, "email", "") or "").lower() == email), None)
        if not match:
            print(f"  ❌  No Supabase user with email {email}")
            print(f"     ({len(users)} total users in the project)")
            sys.exit(1)
        user_id = match.id
        print(f"  ✓ user_id = {user_id}")
    except Exception as e:
        print(f"  ❌  Supabase user lookup failed: {e}")
        sys.exit(1)

    # ---- Find the active subscription in Stripe ----
    print(f"\nLooking up Stripe customers with email {email} ...")
    customers = stripe.Customer.list(email=email, limit=10).data
    if not customers:
        print(f"  ❌  No Stripe customers found for {email}.")
        sys.exit(1)
    print(f"  Found {len(customers)} customer(s).")

    # For each customer, look for an active subscription. Use the most recent one.
    active_sub = None
    for cust in customers:
        subs = stripe.Subscription.list(customer=cust.id, status="active", limit=5).data
        for s in subs:
            if active_sub is None or s.created > active_sub.created:
                active_sub = s
    if active_sub is None:
        # Try any subscription, not just active
        for cust in customers:
            subs = stripe.Subscription.list(customer=cust.id, limit=5).data
            for s in subs:
                if active_sub is None or s.created > active_sub.created:
                    active_sub = s
    if active_sub is None:
        print("  ❌  No subscriptions found at all for those customer(s).")
        sys.exit(1)
    print(f"  ✓ Found subscription {active_sub.id} (status={active_sub.status})")

    # ---- Build the row and write to Supabase ----
    # Stripe-py v10+ objects don't support .get(); use bracket or attribute access.
    def sf(obj, key, default=None):
        """Safe field access for both Stripe objects and plain dicts."""
        try:
            return obj[key]
        except (KeyError, TypeError):
            return default

    period_end_ts = sf(active_sub, "current_period_end")
    if not period_end_ts:
        # current_period_end moved onto items.data[0] in newer Stripe API.
        try:
            period_end_ts = active_sub["items"]["data"][0]["current_period_end"]
        except (KeyError, IndexError, TypeError):
            period_end_ts = None
    payload = {
        "user_id": str(user_id),
        "stripe_customer_id": sf(active_sub, "customer"),
        "stripe_subscription_id": sf(active_sub, "id"),
        "status": sf(active_sub, "status"),
        "current_period_end": (
            datetime.utcfromtimestamp(period_end_ts).isoformat() + "Z"
            if period_end_ts else None
        ),
        "cancel_at_period_end": bool(sf(active_sub, "cancel_at_period_end", False)),
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    print(f"\nWriting to Supabase subscriptions table:")
    for k, v in payload.items():
        print(f"    {k}: {v}")

    try:
        result = supa.table("subscriptions").upsert(
            payload, on_conflict="user_id"
        ).execute()
        print(f"\n✅ Wrote subscription row. Result data: {result.data}")
    except Exception as e:
        print(f"\n❌ Supabase write failed: {e}")
        print("\nIf the error mentions 'permission' or 'rls', your SUPABASE_SERVICE_KEY")
        print("is the wrong key (probably the anon/publishable one). Copy the 'secret'")
        print("key from Supabase → Project Settings → API Keys instead.")
        sys.exit(1)

    print("\nRefresh localhost:8000 in your browser — you should be unlocked.")


if __name__ == "__main__":
    main()
