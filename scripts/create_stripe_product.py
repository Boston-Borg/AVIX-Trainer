"""
One-time setup: create the AVX product + $20/month price in Stripe.

Run after you've put STRIPE_SECRET_KEY in your .env file:

    python scripts/create_stripe_product.py

It prints the new price_xxx ID. Copy that into .env as STRIPE_PRICE_ID.

You can run this safely more than once — it always creates a *new* product
and price, so if you re-run it you'll get a fresh price ID. (Stripe doesn't
have a "create or update" for products, so we just leave any old ones in the
dashboard. You can archive them later if you care.)
"""

import os
import sys

# Make sure we can import from the project root regardless of cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dotenv import load_dotenv
load_dotenv()

import stripe

key = os.environ.get("STRIPE_SECRET_KEY")
if not key:
    print("ERROR: STRIPE_SECRET_KEY is not set in your .env file.")
    print("Get it from https://dashboard.stripe.com → Developers → API keys")
    print("Make sure you're in TEST MODE (toggle top-right of the dashboard).")
    sys.exit(1)

if not key.startswith(("sk_test_", "sk_live_")):
    print(f"ERROR: STRIPE_SECRET_KEY looks wrong (starts with: {key[:8]}...).")
    print("Expected something like sk_test_... or sk_live_...")
    sys.exit(1)

if key.startswith("sk_live_"):
    confirm = input(
        "⚠️  You're using a LIVE Stripe key — this will create a real product.\n"
        "    Type 'yes' to continue: "
    )
    if confirm.strip().lower() != "yes":
        print("Cancelled.")
        sys.exit(0)

stripe.api_key = key

print("Creating product 'AVX — PPL Study Companion' ...")
product = stripe.Product.create(
    name="AVX — PPL Study Companion",
    description=(
        "AI-powered Private Pilot License study companion. CFI chat, FAA-cited "
        "study guides, flashcards, quizzes, and a DPE oral exam simulator."
    ),
)
print(f"  → product.id = {product.id}")

print("Creating price: $20/month recurring ...")
price = stripe.Price.create(
    product=product.id,
    unit_amount=2000,      # cents
    currency="usd",
    recurring={"interval": "month"},
)
print(f"  → price.id = {price.id}")

print()
print("=" * 70)
print("✅ Done. Paste this line into your .env file:")
print()
print(f"    STRIPE_PRICE_ID={price.id}")
print()
print("=" * 70)
