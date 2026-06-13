-- =====================================================================
--  AVIX — Saved Aircraft migration
--
--  Adds the saved_aircraft table backing the Aircraft Lookup feature:
--  each row is one FAA N-number lookup a user chose to bookmark, with
--  the full FAA registry result stored as jsonb.
--
--  HOW TO RUN THIS:
--    1. Open https://supabase.com/dashboard
--    2. Pick your AVIX project
--    3. Left sidebar → SQL Editor → "New query"
--    4. Paste this entire file in, click "Run"
--    5. You should see "Success. No rows returned" (or similar).
--
--  Idempotent. Safe to re-run.
-- =====================================================================


-- ---------------------------------------------------------------------
-- SAVED_AIRCRAFT — per-user bookmarks of FAA registry lookups.
--   data holds the full lookup JSON (manufacturer, model, owner, ...)
--   so the saved list renders without re-scraping the FAA site.
--   (user_id, n_number) is unique: re-saving the same tail number
--   simply refreshes the stored data (the server upserts on conflict).
-- ---------------------------------------------------------------------
create table if not exists public.saved_aircraft (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users(id) on delete cascade,
  n_number    text not null,
  data        jsonb,
  created_at  timestamptz not null default now(),
  unique (user_id, n_number)
);

-- Fast "list my aircraft" reads.
create index if not exists saved_aircraft_user_created_idx
  on public.saved_aircraft (user_id, created_at desc);

-- RLS: the Flask server talks to this table with the service_role key and
-- scopes every query to the caller's user_id, but we still enable RLS with
-- owner-only policies so direct PostgREST/browser access can't leak rows —
-- same defense-in-depth pattern as user_preferences / chat_sessions.
alter table public.saved_aircraft enable row level security;

drop policy if exists "users read own saved aircraft" on public.saved_aircraft;
create policy "users read own saved aircraft"
  on public.saved_aircraft for select using (auth.uid() = user_id);

drop policy if exists "users insert own saved aircraft" on public.saved_aircraft;
create policy "users insert own saved aircraft"
  on public.saved_aircraft for insert with check (auth.uid() = user_id);

drop policy if exists "users update own saved aircraft" on public.saved_aircraft;
create policy "users update own saved aircraft"
  on public.saved_aircraft for update using (auth.uid() = user_id);

drop policy if exists "users delete own saved aircraft" on public.saved_aircraft;
create policy "users delete own saved aircraft"
  on public.saved_aircraft for delete using (auth.uid() = user_id);
