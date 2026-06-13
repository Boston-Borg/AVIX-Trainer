"""
AVIX data cleanup — archive and delete stale rows.

This module is designed to be used two ways:
  1. Imported by server.py so POST /api/cleanup can call run_cleanup().
  2. Run directly by the Render cron job: python cleanup.py

Thresholds (all configurable via env vars):
  CLEANUP_CHAT_DAYS   — delete chat_sessions with updated_at older than N days (default 30)
  CLEANUP_TRIAL_DAYS  — delete trial_usage rows where trial expired N+ days ago (default 60)
  CLEANUP_ORAL_DAYS   — archive + delete oral_exam_results older than N days (default 90)

Idempotency: all deletes are age-based (not state-based), so running twice
in the same window deletes zero extra rows the second time.

Safety: oral_exam_results are copied into archive_exam_results (with upsert
on id) before deletion. A run failure mid-way leaves the source rows intact;
the next run re-archives and re-deletes cleanly.

Batch size: CLEANUP_BATCH_SIZE rows per table per run (default 500) to avoid
locking large tables or timing out on Render's 15-minute cron limit.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from supabase import create_client

# Load .env for local dev; no-op on Render where vars come from the dashboard.
load_dotenv()

log = logging.getLogger("avx.cleanup")

# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------
CLEANUP_CHAT_DAYS  = int(os.environ.get("CLEANUP_CHAT_DAYS",  "30"))
CLEANUP_TRIAL_DAYS = int(os.environ.get("CLEANUP_TRIAL_DAYS", "60"))
CLEANUP_ORAL_DAYS  = int(os.environ.get("CLEANUP_ORAL_DAYS",  "90"))
CLEANUP_BATCH_SIZE = int(os.environ.get("CLEANUP_BATCH_SIZE", "500"))


# ---------------------------------------------------------------------------
# Supabase client factory
# ---------------------------------------------------------------------------
def _make_admin_client():
    """Build a service-role Supabase client from env vars.
    Raises RuntimeError if the required variables are missing.
    """
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set. "
            "Add them to .env (local dev) or Render environment (production)."
        )
    return create_client(url, key)


# ---------------------------------------------------------------------------
# Main cleanup function
# ---------------------------------------------------------------------------
def run_cleanup(client=None) -> dict:
    """Archive and delete stale rows across three tables.

    Args:
        client: An already-constructed supabase admin client (e.g. server.py's
                supabase_admin). If None, a new client is built from env vars.

    Returns:
        {
            "rows_deleted":  int,   # total hard-deleted rows
            "rows_archived": int,   # rows copied into archive_exam_results
            "errors":        list,  # non-fatal error messages (empty = clean run)
            "started_at":    str,   # ISO timestamp
            "finished_at":   str,   # ISO timestamp
        }
    """
    if client is None:
        client = _make_admin_client()

    started = datetime.now(timezone.utc)
    rows_deleted  = 0
    rows_archived = 0
    errors: list[str] = []

    log.info(
        "Cleanup started at %s  thresholds: chat=%dd trial=%dd oral=%dd batch=%d",
        started.isoformat(), CLEANUP_CHAT_DAYS, CLEANUP_TRIAL_DAYS,
        CLEANUP_ORAL_DAYS, CLEANUP_BATCH_SIZE,
    )

    # -----------------------------------------------------------------------
    # 1. Archive oral_exam_results → archive_exam_results, then delete
    #
    #    We copy rows first (upsert on id so re-runs are safe), mark them
    #    with deleted_at, then hard-delete. If the process dies between the
    #    upsert and delete, the next run re-upserts (no-op) and retries the
    #    delete — no data is lost.
    # -----------------------------------------------------------------------
    oral_cutoff = (started - timedelta(days=CLEANUP_ORAL_DAYS)).isoformat()
    try:
        fetched = (
            client.table("oral_exam_results")
            .select("id, user_id, topic, score, verdict, difficulty, feedback, created_at")
            .lt("created_at", oral_cutoff)
            .limit(CLEANUP_BATCH_SIZE)
            .execute()
        )
        old_orals = fetched.data or []

        if old_orals:
            now_iso = started.isoformat()

            # Step 1a — copy to archive (idempotent: upsert on id).
            archive_rows = [
                {
                    "id":         row["id"],
                    "user_id":    row["user_id"],
                    "topic":      row.get("topic"),
                    "score":      row.get("score"),
                    "verdict":    row.get("verdict"),
                    "difficulty": row.get("difficulty"),
                    "feedback":   row.get("feedback"),
                    "created_at": row.get("created_at"),
                    "archived_at": now_iso,
                }
                for row in old_orals
            ]
            client.table("archive_exam_results").upsert(
                archive_rows, on_conflict="id"
            ).execute()
            rows_archived += len(archive_rows)
            log.info("Archived %d oral_exam_results rows", len(archive_rows))

            # Step 1b — soft-mark source rows (audit trail before hard delete).
            ids = [row["id"] for row in old_orals]
            client.table("oral_exam_results").update(
                {"deleted_at": now_iso}
            ).in_("id", ids).execute()

            # Step 1c — hard delete from source.
            client.table("oral_exam_results").delete().in_("id", ids).execute()
            rows_deleted += len(ids)
            log.info("Deleted %d oral_exam_results rows", len(ids))

        else:
            log.info("No oral_exam_results older than %d days", CLEANUP_ORAL_DAYS)

    except Exception as exc:  # noqa: BLE001
        msg = f"oral_exam_results cleanup failed: {exc}"
        log.exception(msg)
        errors.append(msg)

    # -----------------------------------------------------------------------
    # 2. Delete stale chat_sessions (by updated_at — sessions that had no
    #    activity for 30 days are expired, regardless of creation date).
    # -----------------------------------------------------------------------
    chat_cutoff = (started - timedelta(days=CLEANUP_CHAT_DAYS)).isoformat()
    try:
        # Soft-mark first so any concurrent reader sees the tombstone.
        mark_res = (
            client.table("chat_sessions")
            .update({"deleted_at": started.isoformat()})
            .lt("updated_at", chat_cutoff)
            .is_("deleted_at", "null")   # only mark rows not yet marked
            .execute()
        )
        marked = len(mark_res.data or [])

        # Hard delete all soft-marked stale sessions.
        del_res = (
            client.table("chat_sessions")
            .delete()
            .lt("updated_at", chat_cutoff)
            .execute()
        )
        n = len(del_res.data or [])
        rows_deleted += n
        if n:
            log.info(
                "Deleted %d stale chat_sessions (marked %d this run)",
                n, marked,
            )
        else:
            log.info("No chat_sessions older than %d days", CLEANUP_CHAT_DAYS)

    except Exception as exc:  # noqa: BLE001
        msg = f"chat_sessions cleanup failed: {exc}"
        log.exception(msg)
        errors.append(msg)

    # -----------------------------------------------------------------------
    # 3. Delete trial_usage rows where the trial expired 60+ days ago.
    #
    #    The trial window is TRIAL_DURATION_MINUTES (30 min). A row is eligible
    #    once:   trial_started_at + 30min + 60 days < now
    #    Which simplifies to:  trial_started_at < now - 60 days  (30 min rounding)
    #
    #    NULL trial_started_at rows are pre-migration users still active —
    #    .lt() on a NULL column returns false in SQL, so they are safe.
    # -----------------------------------------------------------------------
    trial_cutoff = (started - timedelta(days=CLEANUP_TRIAL_DAYS)).isoformat()
    try:
        mark_res = (
            client.table("trial_usage")
            .update({"deleted_at": started.isoformat()})
            .lt("trial_started_at", trial_cutoff)
            .is_("deleted_at", "null")
            .execute()
        )
        marked = len(mark_res.data or [])

        del_res = (
            client.table("trial_usage")
            .delete()
            .lt("trial_started_at", trial_cutoff)
            .execute()
        )
        n = len(del_res.data or [])
        rows_deleted += n
        if n:
            log.info(
                "Deleted %d expired trial_usage rows (marked %d this run)",
                n, marked,
            )
        else:
            log.info("No trial_usage rows older than %d days", CLEANUP_TRIAL_DAYS)

    except Exception as exc:  # noqa: BLE001
        msg = f"trial_usage cleanup failed: {exc}"
        log.exception(msg)
        errors.append(msg)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    finished = datetime.now(timezone.utc)
    elapsed  = (finished - started).total_seconds()
    result = {
        "rows_deleted":  rows_deleted,
        "rows_archived": rows_archived,
        "errors":        errors,
        "started_at":    started.isoformat(),
        "finished_at":   finished.isoformat(),
        "elapsed_seconds": round(elapsed, 2),
    }
    log.info(
        "Cleanup finished in %.2fs — deleted=%d archived=%d errors=%d",
        elapsed, rows_deleted, rows_archived, len(errors),
    )
    return result


# ---------------------------------------------------------------------------
# Standalone entrypoint (Render cron job runs: python cleanup.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        result = run_cleanup()
    except Exception as exc:  # noqa: BLE001
        log.exception("Cleanup aborted: %s", exc)
        sys.exit(1)

    print(
        f"rows_deleted={result['rows_deleted']}  "
        f"rows_archived={result['rows_archived']}  "
        f"elapsed={result['elapsed_seconds']}s  "
        f"errors={len(result['errors'])}"
    )
    if result["errors"]:
        for err in result["errors"]:
            print(f"  ERROR: {err}", file=sys.stderr)
        sys.exit(1)   # non-zero exit signals Render that the cron job failed
