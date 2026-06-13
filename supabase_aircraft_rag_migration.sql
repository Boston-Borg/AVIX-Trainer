-- =====================================================================
--  AVIX — Aircraft RAG cache migration
--
--  Adds a processed_at timestamp to saved_aircraft. The server stamps this
--  column once an aircraft's facts have been embedded into the local vector
--  index (retrieval.embed_aircraft). The frontend polls /api/aircraft_status
--  and shows "Processing…" until processed_at is set, then "Ready".
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

-- processed_at: NULL = embeddings not built yet; timestamp = cache is warm.
alter table public.saved_aircraft
  add column if not exists processed_at timestamptz;
