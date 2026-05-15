"""
One-time setup: create a 100%-off Stripe coupon + a promo code for AVX creators.

Run this once after Stripe is configured:

    python scripts/create_stripe_promo.py

By default it creates a promotion code "AVXCREW" backed by a 100%-off-forever
coupon. Anyone who types AVXCREW at checkout subscribes for $0/month with no
expiration.

To change the code or limit who can use it, pass arguments:

    python scripts/create_stripe_promo.py --code MYCODE --max-redemptions 5
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dotenv import load_dotenv
load_dotenv()

import stripe

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code",
        default="AVXCREW",
        help="The text users type at checkout (default: AVXCREW)",
    )
    parser.add_argument(
        "--percent-off",
        type=int,
        default=100,
        help="Discount percentage 1-100 (default: 100 for free access)",
    )
    parser.add_argument(
        "--max-redemptions",
        type=int,
        default=None,
        help="Maximum total uses of this promo code (default: unlimited)",
    )
    parser.add_argument(
        "--duration",
        default="forever",
        choices=["forever", "once", "repeating"],
        help="How long the discount lasts on the subscription (default: forever)",
    )
    args = parser.parse_args()

    key = os.environ.get("STRIPE_SECRET_KEY")
    if not key:
        print("ERROR: STRIPE_SECRET_KEY is not set in your .env file.")
        sys.exit(1)
    if not key.startswith(("sk_test_", "sk_live_")):
        print(f"ERROR: STRIPE_SECRET_KEY looks wrong (starts with: {key[:8]}...).")
        sys.exit(1)
    if key.startswith("sk_live_"):
        confirm = input("⚠️  LIVE mode key detected — creating a real promo code. Type 'yes': ")
        if confirm.strip().lower() != "yes":
            print("Cancelled.")
            sys.exit(0)

    stripe.api_key = key
    code = args.code.upper()

    # ----- Step 1: create the coupon (the discount rule) -----
    print(f"Creating coupon: {args.percent_off}% off, duration={args.duration} ...")
    coupon_kwargs = {
        "percent_off": args.percent_off,
        "duration": args.duration,
        "name": f"AVX Creator Discount ({args.percent_off}% off)",
    }
    if args.duration == "repeating":
        # Stripe requires duration_in_months for "repeating"
        coupon_kwargs["duration_in_months"] = 12
    coupon = stripe.Coupon.create(**coupon_kwargs)
    print(f"  → coupon.id = {coupon.id}")

    # ----- Step 2: create the promotion code that maps to the coupon -----
    print(f"Creating promotion code: {code} ...")
    promo_kwargs = {
        "coupon": coupon.id,
        "code": code,
    }
    if args.max_redemptions is not None:
        promo_kwargs["max_redemptions"] = args.max_redemptions
    try:
        promo = stripe.PromotionCode.create(**promo_kwargs)
        print(f"  → promotion_code.id = {promo.id}")
    except stripe.error.InvalidRequestError as e:
        msg = str(e)
        if "already exists" in msg.lower() or "uniqueness" in msg.lower():
            print(f"  ⚠️  A promotion code with code={code!r} already exists in Stripe.")
            print(f"     Use a different --code, or delete the old one in the dashboard.")
            sys.exit(1)
        raise

    print()
    print("=" * 70)
    print(f"✅ Done. Anyone can now type this at checkout to subscribe free:")
    print()
    print(f"    {code}")
    print()
    if args.max_redemptions:
        print(f"   ({args.max_redemptions} total uses, then it expires.)")
    else:
        print(f"   (Unlimited uses. To revoke later: Stripe dashboard → Products →")
        print(f"    Coupons → find this one → Archive.)")
    print("=" * 70)


if __name__ == "__main__":
    main()
