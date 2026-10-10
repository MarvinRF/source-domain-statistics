"""Profile summarization pipeline: LLM summaries per user, then embedding.

Invoked through ``python -m acw.embeddings.profile_summarization.cli``.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import UTC, datetime

from .cache import SummaryCache
from .config import (
    LLM_CONCURRENCY,
    MAX_BOOKS_PROFILE,
    MIN_H4_ENTITIES,
    PROCESSED_DIR,
    TEMPERATURE_H3,
    TEMPERATURE_H3_CONCRETE,
    TEMPERATURE_H4,
    TEMPERATURE_H4_CONCRETE,
)
from .embedder import embed_profiles
from .gte_encoder import GTEEncoder
from .llm_client import LLMClient, build_client
from .loader import (
    domain_from_heuristic,
    load_book_history,
    load_book_intent_cache,
    load_book_metadata,
    load_test_users,
)
from .metadata import save_metadata, tagged_stem
from .prompts import (
    build_h3_concrete_prompt,
    build_h3_games_prompt,
    build_h3_movies_prompt,
    build_h3_prompt,
    build_h4_concrete_prompt,
    build_h4_games_prompt,
    build_h4_movies_prompt,
    build_h4_prompt,
)
from .tracer import save_debug_trace
from .validation import validate_summary

log = logging.getLogger(__name__)


def _build_prompt(
    heuristic: str,
    recent_asins: list[str],
    book_meta: dict,
    book_intent: dict | None,
) -> tuple[str | None, float | None]:
    """Returns (prompt, temperature), or (None, None) if this user should be
    skipped (cached as None) — e.g. no book metadata / not enough intent
    entities for this heuristic."""
    h3_builders = {
        "h3": (build_h3_prompt, TEMPERATURE_H3),
        "h3_concrete": (build_h3_concrete_prompt, TEMPERATURE_H3_CONCRETE),
        "h3_movies": (build_h3_movies_prompt, TEMPERATURE_H3),
        "h3_games": (build_h3_games_prompt, TEMPERATURE_H3),
    }
    h4_builders = {
        "h4": (build_h4_prompt, TEMPERATURE_H4),
        "h4_concrete": (build_h4_concrete_prompt, TEMPERATURE_H4_CONCRETE),
        "h4_movies": (build_h4_movies_prompt, TEMPERATURE_H4),
        "h4_games": (build_h4_games_prompt, TEMPERATURE_H4),
    }

    if heuristic in h3_builders:
        books_data = [book_meta[a] for a in recent_asins if a in book_meta]
        if not books_data:
            return None, None
        builder, temperature = h3_builders[heuristic]
        return builder(books_data), temperature

    if heuristic in h4_builders:
        if not book_intent:
            return None, None
        entities = [
            book_intent[a]
            for a in recent_asins
            if a in book_intent and not book_intent[a].get("_fallback")
        ]
        if len(entities) < MIN_H4_ENTITIES:
            return None, None
        builder, temperature = h4_builders[heuristic]
        return builder(entities), temperature

    return None, None


async def summarize_users_async(
    heuristic: str,
    test_users: list[str],
    book_history: dict,
    book_meta: dict,
    book_intent: dict | None,
    client: LLMClient,
    dry_run: bool,
    limit: int | None,
    run_tag: str | None = None,
) -> dict:
    cache_path = PROCESSED_DIR / f"{tagged_stem(heuristic, run_tag)}.json"
    cache = SummaryCache(cache_path)

    pending = [u for u in test_users if u not in cache]
    if limit:
        pending = pending[:limit]

    log.info(f"{heuristic.upper()} pending users: {len(pending):,} (concurrency={LLM_CONCURRENCY})")

    if not pending:
        return dict(cache.items())

    t_start = time.time()
    progress = {"completed": 0}
    semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

    async def process_one(uid: str):
        recent_asins = book_history.get(uid, [])[-MAX_BOOKS_PROFILE:]
        prompt, temperature = _build_prompt(heuristic, recent_asins, book_meta, book_intent)

        if prompt is None:
            output, elapsed = None, 0.0
        else:
            t0 = time.time()
            async with semaphore:
                try:
                    output = await client.agenerate(prompt, temperature)
                    elapsed = time.time() - t0
                    if not validate_summary(output, heuristic):
                        output = None
                except Exception as exc:
                    log.warning(f"{heuristic.upper()} failed for {uid}: {exc}")
                    output = None
                    elapsed = time.time() - t0

            save_debug_trace(
                heuristic,
                uid,
                prompt,
                output,
                {
                    "elapsed_seconds": round(elapsed, 3),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "temperature": temperature,
                    "model": client.model,
                },
            )

        if not dry_run:
            # asyncio is single-threaded/cooperative — concurrent coroutines
            # mutating the same cache dict here is safe, no lock needed.
            cache.set(uid, output)
        elif output:
            print(f"\n  [{heuristic.upper()}] {uid[:24]}...")
            print(f"  {output[:300]}")

        progress["completed"] += 1
        if progress["completed"] % 50 == 0:
            elapsed_total = (time.time() - t_start) / 60
            rate = progress["completed"] / (time.time() - t_start)
            eta = (len(pending) - progress["completed"]) / rate / 60 if rate > 0 else float("nan")
            log.info(
                f"  [{heuristic.upper()}] {progress['completed']:,}/{len(pending):,}  "
                f"elapsed={elapsed_total:.0f}min  ETA={eta:.0f}min"
            )

    await asyncio.gather(*(process_one(uid) for uid in pending))

    if not dry_run:
        cache.flush()

    return dict(cache.items())


async def async_main(args):
    log.info("=" * 80)
    log.info("Cross-Domain Profile Summarization")
    log.info("=" * 80)
    log.info(f"LLM model       : {args.model}")
    log.info(f"Embedding model : {args.embedding_model}")
    log.info(f"Heuristic       : {args.heuristic}")
    log.info(f"Dry run         : {args.dry_run}")
    run_tag = getattr(args, "run_tag", None)
    log.info(f"Run tag         : {run_tag or '(none — untagged filenames)'}")

    embed_only = getattr(args, "embed_only", False)
    client = None
    if not embed_only:
        client = build_client(args.model)
        log.info("\nChecking LLM backend...")
        if not client.health_check():
            import os

            backend = os.getenv("LLM_BACKEND", "ollama")
            if backend == "ollama":
                log.error("Ollama not reachable. Run: ollama serve")
            else:
                log.error(
                    "vLLM server not reachable. Start it with: python -m vllm.entrypoints.openai.api_server"
                )
            sys.exit(1)
        log.info("  LLM backend OK ✓")
    else:
        log.info("  --embed-only: skipping LLM backend check.")

    domain = domain_from_heuristic(args.heuristic)
    log.info(f"Domain          : {domain or 'music (default)'}")

    test_users = load_test_users(
        all_users=getattr(args, "all_users", False),
        user_list_path=getattr(args, "user_list", None),
        domain=domain,
    )
    book_history = load_book_history(domain=domain)

    run_h3 = args.heuristic in ("h3", "both")
    run_h4 = args.heuristic in ("h4", "both")
    run_h3_concrete = args.heuristic == "h3_concrete"
    run_h4_concrete = args.heuristic == "h4_concrete"
    run_h3_movies = args.heuristic == "h3_movies"
    run_h4_movies = args.heuristic == "h4_movies"
    run_h3_games = args.heuristic == "h3_games"
    run_h4_games = args.heuristic == "h4_games"

    needs_meta = run_h3 or run_h3_concrete or run_h3_movies or run_h3_games
    needs_intent = run_h4 or run_h4_concrete or run_h4_movies or run_h4_games

    book_meta = load_book_metadata(domain=domain) if (needs_meta and not embed_only) else {}
    book_intent = load_book_intent_cache() if (needs_intent and not embed_only) else None

    text_only = getattr(args, "text_only", False)

    device = getattr(args, "device", "cpu")
    if not text_only:
        log.info(f"\nLoading embedding model on {device}...")
        if args.embedding_model == "Alibaba-NLP/gte-large-en-v1.5":
            # GTEEncoder works around the RoPE position-buffer issue in this
            # model's remote code (see gte_encoder.py).
            sbert = GTEEncoder(args.embedding_model, device=device)
        else:
            from sentence_transformers import SentenceTransformer

            sbert = SentenceTransformer(args.embedding_model, device=device, trust_remote_code=True)
    else:
        sbert = None
        log.info("\n--text-only: embedding step will be skipped.")

    # ── H3 ────────────────────────────────────────────────────────────────
    if run_h3:
        log.info("\n" + "─" * 80)
        log.info("H3 — RAW SYNOPSES (evocative)")
        log.info("─" * 80)
        if not embed_only:
            h3_summaries = await summarize_users_async(
                "h3",
                test_users,
                book_history,
                book_meta,
                None,
                client,
                args.dry_run,
                args.limit,
                run_tag,
            )
        else:
            cache_path = PROCESSED_DIR / f"{tagged_stem('h3', run_tag)}.json"
            h3_summaries = dict(SummaryCache(cache_path).items())
            log.info(
                f"  --embed-only: loaded {len(h3_summaries):,} cached H3 summaries from {cache_path.name}"
            )
        if not args.dry_run:
            if text_only:
                save_metadata(
                    "h3",
                    h3_summaries,
                    None,
                    args.model,
                    args.embedding_model,
                    run_tag=run_tag,
                    text_only=True,
                )
            else:
                emb = embed_profiles(h3_summaries, test_users, "h3", sbert, False, run_tag=run_tag)
                save_metadata(
                    "h3",
                    h3_summaries,
                    emb.shape,
                    args.model,
                    args.embedding_model,
                    run_tag=run_tag,
                )

    # ── H4 ────────────────────────────────────────────────────────────────
    if run_h4:
        log.info("\n" + "─" * 80)
        log.info("H4 — INTENT ENTITIES (evocative)")
        log.info("─" * 80)
        if not embed_only:
            h4_summaries = await summarize_users_async(
                "h4",
                test_users,
                book_history,
                {},
                book_intent,
                client,
                args.dry_run,
                args.limit,
                run_tag,
            )
        else:
            cache_path = PROCESSED_DIR / f"{tagged_stem('h4', run_tag)}.json"
            h4_summaries = dict(SummaryCache(cache_path).items())
            log.info(
                f"  --embed-only: loaded {len(h4_summaries):,} cached H4 summaries from {cache_path.name}"
            )
        if not args.dry_run:
            if text_only:
                save_metadata(
                    "h4",
                    h4_summaries,
                    None,
                    args.model,
                    args.embedding_model,
                    run_tag=run_tag,
                    text_only=True,
                )
            else:
                emb = embed_profiles(h4_summaries, test_users, "h4", sbert, False, run_tag=run_tag)
                save_metadata(
                    "h4",
                    h4_summaries,
                    emb.shape,
                    args.model,
                    args.embedding_model,
                    run_tag=run_tag,
                )

    # ── H3_concrete ───────────────────────────────────────────────────────
    if run_h3_concrete:
        log.info("\n" + "─" * 80)
        log.info("H3_CONCRETE — RAW SYNOPSES (factual music vocabulary)")
        log.info("─" * 80)
        h3c_summaries = await summarize_users_async(
            "h3_concrete",
            test_users,
            book_history,
            book_meta,
            None,
            client,
            args.dry_run,
            args.limit,
            run_tag,
        )
        if not args.dry_run:
            emb = embed_profiles(
                h3c_summaries, test_users, "h3_concrete", sbert, False, run_tag=run_tag
            )
            save_metadata(
                "h3_concrete",
                h3c_summaries,
                emb.shape,
                args.model,
                args.embedding_model,
                run_tag=run_tag,
            )

    # ── H4_concrete ───────────────────────────────────────────────────────
    if run_h4_concrete:
        log.info("\n" + "─" * 80)
        log.info("H4_CONCRETE — INTENT ENTITIES (factual music vocabulary)")
        log.info("─" * 80)
        h4c_summaries = await summarize_users_async(
            "h4_concrete",
            test_users,
            book_history,
            {},
            book_intent,
            client,
            args.dry_run,
            args.limit,
            run_tag,
        )
        if not args.dry_run:
            emb = embed_profiles(
                h4c_summaries, test_users, "h4_concrete", sbert, False, run_tag=run_tag
            )
            save_metadata(
                "h4_concrete",
                h4c_summaries,
                emb.shape,
                args.model,
                args.embedding_model,
                run_tag=run_tag,
            )

    # ── H3_movies ─────────────────────────────────────────────────────────
    if run_h3_movies:
        log.info("\n" + "─" * 80)
        log.info("H3_MOVIES — RAW SYNOPSES (cinematic register)")
        log.info("─" * 80)
        h3m_summaries = await summarize_users_async(
            "h3_movies",
            test_users,
            book_history,
            book_meta,
            None,
            client,
            args.dry_run,
            args.limit,
            run_tag,
        )
        if not args.dry_run:
            if text_only:
                save_metadata(
                    "h3_movies",
                    h3m_summaries,
                    None,
                    args.model,
                    args.embedding_model,
                    run_tag=run_tag,
                    text_only=True,
                )
            else:
                emb = embed_profiles(
                    h3m_summaries, test_users, "h3_movies", sbert, False, run_tag=run_tag
                )
                save_metadata(
                    "h3_movies",
                    h3m_summaries,
                    emb.shape,
                    args.model,
                    args.embedding_model,
                    run_tag=run_tag,
                )

    # ── H4_movies ─────────────────────────────────────────────────────────
    if run_h4_movies:
        log.info("\n" + "─" * 80)
        log.info("H4_MOVIES — INTENT ENTITIES (cinematic register)")
        log.info("─" * 80)
        h4m_summaries = await summarize_users_async(
            "h4_movies",
            test_users,
            book_history,
            {},
            book_intent,
            client,
            args.dry_run,
            args.limit,
            run_tag,
        )
        if not args.dry_run:
            if text_only:
                save_metadata(
                    "h4_movies",
                    h4m_summaries,
                    None,
                    args.model,
                    args.embedding_model,
                    run_tag=run_tag,
                    text_only=True,
                )
            else:
                emb = embed_profiles(
                    h4m_summaries, test_users, "h4_movies", sbert, False, run_tag=run_tag
                )
                save_metadata(
                    "h4_movies",
                    h4m_summaries,
                    emb.shape,
                    args.model,
                    args.embedding_model,
                    run_tag=run_tag,
                )

    # ── H3_games ──────────────────────────────────────────────────────────
    if run_h3_games:
        log.info("\n" + "─" * 80)
        log.info("H3_GAMES — RAW SYNOPSES (ludic register)")
        log.info("─" * 80)
        h3g_summaries = await summarize_users_async(
            "h3_games",
            test_users,
            book_history,
            book_meta,
            None,
            client,
            args.dry_run,
            args.limit,
            run_tag,
        )
        if not args.dry_run:
            if text_only:
                save_metadata(
                    "h3_games",
                    h3g_summaries,
                    None,
                    args.model,
                    args.embedding_model,
                    run_tag=run_tag,
                    text_only=True,
                )
            else:
                emb = embed_profiles(
                    h3g_summaries, test_users, "h3_games", sbert, False, run_tag=run_tag
                )
                save_metadata(
                    "h3_games",
                    h3g_summaries,
                    emb.shape,
                    args.model,
                    args.embedding_model,
                    run_tag=run_tag,
                )

    # ── H4_games ──────────────────────────────────────────────────────────
    if run_h4_games:
        log.info("\n" + "─" * 80)
        log.info("H4_GAMES — INTENT ENTITIES (ludic register)")
        log.info("─" * 80)
        h4g_summaries = await summarize_users_async(
            "h4_games",
            test_users,
            book_history,
            {},
            book_intent,
            client,
            args.dry_run,
            args.limit,
            run_tag,
        )
        if not args.dry_run:
            if text_only:
                save_metadata(
                    "h4_games",
                    h4g_summaries,
                    None,
                    args.model,
                    args.embedding_model,
                    run_tag=run_tag,
                    text_only=True,
                )
            else:
                emb = embed_profiles(
                    h4g_summaries, test_users, "h4_games", sbert, False, run_tag=run_tag
                )
                save_metadata(
                    "h4_games",
                    h4g_summaries,
                    emb.shape,
                    args.model,
                    args.embedding_model,
                    run_tag=run_tag,
                )

    log.info("\n" + "=" * 80)
    log.info("DONE")
    log.info("=" * 80)
    log.info("\nNext step:")
    log.info("  python -m acw.evaluation.evaluate --all")
