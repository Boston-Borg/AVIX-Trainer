-- =====================================================================
--  AVIX — Cleanup migration
--
--  Adds deleted_at audit columns to the three tables that the cleanup
--  job manages, and creates the archive_exam_results table that oral
--  exam records are copied into before deletion.
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


-- ---------------------------------------------------------------------
-- 1. Add deleted_at to chat_sessions
--    Soft-delete marker; set by the cleanup job just before archiving
--    or deletion so in-flight reads can detect the row is going away.
-- ---------------------------------------------------------------------
alter table public.chat_sessions
  add column if not exists deleted_at timestamptz;

create index if not exists chat_sessions_deleted_at_idx
  on public.chat_sessions (deleted_at)
  where deleted_at is not null;


-- ---------------------------------------------------------------------
-- 2. Add deleted_at to trial_usage
-- ---------------------------------------------------------------------
alter table public.trial_usage
  add column if not exists deleted_at timestamptz;

create index if not exists trial_usage_deleted_at_idx
  on public.trial_usage (deleted_at)
  where deleted_at is not null;


-- ---------------------------------------------------------------------
-- 3. Add deleted_at to oral_exam_results
-- ---------------------------------------------------------------------
alter table public.oral_exam_results
  add column if not exists deleted_at timestamptz;

create index if not exists oral_exam_results_deleted_at_idx
  on public.oral_exam_results (deleted_at)
  where deleted_at is not null;


-- ---------------------------------------------------------------------
-- 4. archive_exam_results — identical schema to oral_exam_results plus
--    an archived_at timestamp. id is kept so upsert is idempotent: if
--    cleanup runs twice before the source row is deleted the second
--    upsert just refreshes the archived_at on the existing archive row.
--
--    RLS: enabled, but no user-facing read policy — rows are owned by
--    the server (service_role). Add a select policy here if you ever
--    want to surface archived orals in a user-facing "history" view.
-- ---------------------------------------------------------------------
create table if not exists public.archive_exam_results (
  id           bigint primary key,     -- same id as the original oral_exam_results row
  user_id      uuid not null,
  topic        text,
  score        integer,
  verdict      text,
  difficulty   text,
  feedback     jsonb,
  created_at   timestamptz,
  archived_at  timestamptz not null default now()
);

create index if not exists archive_exam_results_user_idx
  on public.archive_exam_results (user_id, created_at desc);

create index if not exists archive_exam_results_archived_at_idx
  on public.archive_exam_results (archived_at desc);

alter table public.archive_exam_results enable row level security;

-- No user-readable policy by default. The service_role key (used by
-- cleanup.py and the Flask admin client) bypasses RLS entirely.
-- Uncomment the block below if you later want users to see their archive:
--
-- drop policy if exists "users read own archived exams" on public.archive_exam_results;
-- create policy "users read own archived exams"
--   on public.archive_exam_results for select
--   using (auth.uid() = user_id);
