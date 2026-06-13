-- =====================================================================
--  AVIX — Trial v2 migration
--
--  Adds a unified trial_started_at column to trial_usage.
--  The 30-minute trial clock now runs from account creation, not from
--  first feature use.
--
--  HOW TO RUN:
--    1. Open https://supabase.com/dashboard
--    2. Pick your AVIX project
--    3. Left sidebar → SQL Editor → "New query"
--    4. Paste this entire file in, click "Run"
--    5. You should see "Success. No rows returned"
--
--  Idempotent. Safe to re-run.
-- =====================================================================


-- Add trial_started_at to existing trial_usage table.
-- Existing rows get NULL (handled gracefully by the backend; the server
-- will backfill trial_started_at = now() on their next trial check).
alter table public.trial_usage
  add column if not exists trial_started_at timestamptz;

-- Backfill: for any existing row set trial_started_at to the earliest
-- known timestamp (first chat or oral), falling back to now().
-- This gives existing users a fair clock rather than resetting to now().
update public.trial_usage
set trial_started_at = coalesce(
  least(chat_first_at, oral_first_at),
  chat_first_at,
  oral_first_at,
  now()
)
where trial_started_at is null;

-- Index for fast single-row reads by user (already has PK, this is optional).
-- Included for consistency with the other trial_usage query pattern.
create index if not exists trial_usage_started_at_idx
  on public.trial_usage (user_id, trial_started_at);
