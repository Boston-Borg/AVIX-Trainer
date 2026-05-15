-- =====================================================================
--  AVX — Supabase schema for subscriptions, quizzes, and oral exams.
--
--  HOW TO RUN THIS:
--    1. Open https://supabase.com/dashboard
--    2. Pick your AVX project
--    3. Left sidebar → SQL Editor → "New query"
--    4. Paste this entire file in, click "Run"
--    5. You should see "Success. No rows returned" (or similar).
--
--  This is idempotent — safe to run multiple times. It uses
--  `create table if not exists` and replaces policies.
-- =====================================================================


-- ---------------------------------------------------------------------
-- 1. SUBSCRIPTIONS — one row per user, mirrors their Stripe state.
-- ---------------------------------------------------------------------
create table if not exists public.subscriptions (
  user_id                uuid primary key references auth.users(id) on delete cascade,
  stripe_customer_id     text unique,
  stripe_subscription_id text unique,
  status                 text,              -- 'active' | 'trialing' | 'past_due' | 'canceled' | 'incomplete' | ...
  current_period_end     timestamptz,
  cancel_at_period_end   boolean default false,
  updated_at             timestamptz default now()
);

-- Index for webhook lookups by Stripe customer ID.
create index if not exists subscriptions_customer_idx
  on public.subscriptions (stripe_customer_id);

-- Row Level Security: users can READ their own subscription row, but not
-- write it (writes come exclusively from the backend via the service_role key,
-- which bypasses RLS).
alter table public.subscriptions enable row level security;

drop policy if exists "users read own subscription" on public.subscriptions;
create policy "users read own subscription"
  on public.subscriptions
  for select
  using (user_id = auth.uid());


-- ---------------------------------------------------------------------
-- 2. QUIZ RESULTS — every quiz a user takes.
-- ---------------------------------------------------------------------
create table if not exists public.quiz_results (
  id                bigserial primary key,
  user_id           uuid not null references auth.users(id) on delete cascade,
  topic             text not null,
  score             integer not null,
  total             integer not null,
  missed_questions  jsonb,                  -- [{q, choices, correct, picked}, ...]
  created_at        timestamptz default now()
);

create index if not exists quiz_results_user_idx
  on public.quiz_results (user_id, created_at desc);

alter table public.quiz_results enable row level security;

drop policy if exists "users read own quiz results"   on public.quiz_results;
drop policy if exists "users insert own quiz results" on public.quiz_results;
create policy "users read own quiz results"
  on public.quiz_results for select  using (user_id = auth.uid());
create policy "users insert own quiz results"
  on public.quiz_results for insert  with check (user_id = auth.uid());


-- ---------------------------------------------------------------------
-- 3. ORAL EXAM RESULTS — every DPE oral session.
-- ---------------------------------------------------------------------
create table if not exists public.oral_exam_results (
  id           bigserial primary key,
  user_id      uuid not null references auth.users(id) on delete cascade,
  topic        text,
  score        integer,                    -- aggregate 0-100 across questions
  verdict      text,                       -- 'correct' | 'partial' | 'incorrect' | mixed
  difficulty   text,                       -- 'beginner' | 'intermediate' | 'checkride'
  feedback     jsonb,                      -- [{q, answer, verdict, score, feedback}, ...]
  created_at   timestamptz default now()
);

create index if not exists oral_exam_results_user_idx
  on public.oral_exam_results (user_id, created_at desc);

alter table public.oral_exam_results enable row level security;

drop policy if exists "users read own oral results"   on public.oral_exam_results;
drop policy if exists "users insert own oral results" on public.oral_exam_results;
create policy "users read own oral results"
  on public.oral_exam_results for select  using (user_id = auth.uid());
create policy "users insert own oral results"
  on public.oral_exam_results for insert  with check (user_id = auth.uid());
