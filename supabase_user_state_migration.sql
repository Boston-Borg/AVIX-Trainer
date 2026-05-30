-- =====================================================================
--  AVIX — User-state persistence migration
--
--  Adds the tables/columns needed so user preferences, chat sessions, and
--  oral exam history follow the account across devices and reloads.
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
-- 1. USER_PREFERENCES — lightweight per-user settings.
--    One row per user. quick_starts is the configurable CFI chat chip
--    list; topics_viewed is the set of Resources topic IDs the user has
--    opened (drives the green checkmarks in the Resources grid).
-- ---------------------------------------------------------------------
create table if not exists public.user_preferences (
  user_id        uuid primary key references auth.users(id) on delete cascade,
  quick_starts   jsonb not null default '[]'::jsonb,
  topics_viewed  jsonb not null default '[]'::jsonb,
  updated_at     timestamptz default now()
);

alter table public.user_preferences enable row level security;

drop policy if exists "users read own prefs"   on public.user_preferences;
drop policy if exists "users insert own prefs" on public.user_preferences;
drop policy if exists "users update own prefs" on public.user_preferences;
create policy "users read own prefs"
  on public.user_preferences for select using (user_id = auth.uid());
create policy "users insert own prefs"
  on public.user_preferences for insert with check (user_id = auth.uid());
create policy "users update own prefs"
  on public.user_preferences for update using (user_id = auth.uid());


-- ---------------------------------------------------------------------
-- 2. CHAT_SESSIONS — one row per CFI chat conversation.
--    `messages` is the full transcript as a JSONB array; the frontend
--    re-hydrates the chat from this on each load.
-- ---------------------------------------------------------------------
create table if not exists public.chat_sessions (
  id           bigserial primary key,
  user_id      uuid not null references auth.users(id) on delete cascade,
  title        text,
  messages     jsonb not null default '[]'::jsonb,
  created_at   timestamptz default now(),
  updated_at   timestamptz default now()
);

create index if not exists chat_sessions_user_idx
  on public.chat_sessions (user_id, updated_at desc);

alter table public.chat_sessions enable row level security;

drop policy if exists "users read own chats"   on public.chat_sessions;
drop policy if exists "users insert own chats" on public.chat_sessions;
drop policy if exists "users update own chats" on public.chat_sessions;
drop policy if exists "users delete own chats" on public.chat_sessions;
create policy "users read own chats"
  on public.chat_sessions for select using (user_id = auth.uid());
create policy "users insert own chats"
  on public.chat_sessions for insert with check (user_id = auth.uid());
create policy "users update own chats"
  on public.chat_sessions for update using (user_id = auth.uid());
create policy "users delete own chats"
  on public.chat_sessions for delete using (user_id = auth.uid());
