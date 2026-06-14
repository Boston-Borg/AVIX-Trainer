-- =====================================================================
--  AVIX — Terms/Privacy consent record migration
--
--  Stores a tamper-resistant record that a user agreed to the Terms of
--  Service and Privacy Policy at signup: who, when, and which version.
--  Written ONLY by the server (service-role key); users can read their own
--  row but cannot insert/update/delete it, so the record can't be altered
--  from the browser.
--
--  HOW TO RUN: Supabase dashboard → SQL Editor → New query → paste → Run.
--  Idempotent. Safe to re-run.
-- =====================================================================

create table if not exists public.tos_consent (
  user_id      uuid primary key references auth.users(id) on delete cascade,
  accepted_at  timestamptz not null default now(),
  tos_version  text not null,
  created_at   timestamptz not null default now()
);

-- RLS: owner may read their own consent row; no client writes (service-role
-- bypasses RLS, so the server can still insert/update).
alter table public.tos_consent enable row level security;

drop policy if exists "users read own tos consent" on public.tos_consent;
create policy "users read own tos consent"
  on public.tos_consent for select using (auth.uid() = user_id);
