"""H2 profile pipeline: mean-pool each user's item embeddings into one vector."""

from datetime import UTC, datetime

import faiss
import numpy as np

from .config import (
    BOOK_EMBEDDINGS_FILE,
    EMBEDDING_MODEL,
    EXPECTED_DIM,
    MAX_BOOKS_PROFILE,
    N_TEST_USERS,
    PROCESSED_DIR,
    SEED,
)
from .io import load_json, load_user_history, save_json
from .users import load_all_users, sample_test_users
from .vectors import normalize_vectors, validate_embedding_dimensions


def main(all_users: bool = False, domain: str = "music"):

    print("=" * 80)
    print("H2 PROFILE GENERATION")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────────────
    # Load Data
    # ─────────────────────────────────────────────────────────────────────

    print("\nLoading user history...")

    history = load_user_history(domain=domain)

    print(f"Users in history: {len(history):,}")

    print("\nSampling evaluation users...")

    if all_users:
        test_users = load_all_users(domain=domain)
    else:
        test_users = sample_test_users(history)

    print(f"Evaluation users: {len(test_users):,}")

    print("\nLoading embedding index...")

    book_id_to_idx = load_json(PROCESSED_DIR / "book_id_to_idx.json")

    print(f"Indexed books: {len(book_id_to_idx):,}")

    print("\nLoading book embeddings...")

    book_embeddings = np.load(PROCESSED_DIR / BOOK_EMBEDDINGS_FILE).astype(np.float32)

    validate_embedding_dimensions(
        book_embeddings,
        EXPECTED_DIM,
    )

    print(f"Book embeddings shape: {book_embeddings.shape}")

    print("\nNormalizing book embeddings...")

    book_embeddings = normalize_vectors(book_embeddings)

    embedding_dim = book_embeddings.shape[1]

    # ─────────────────────────────────────────────────────────────────────
    # Generate H2 Profiles
    # ─────────────────────────────────────────────────────────────────────

    print("\nGenerating H2 pooled profiles...")

    h2_profiles = []

    valid_users = []

    skipped_users = []

    for uid in test_users:
        user_books = history.get(uid, [])

        recent_books = (
            user_books[-MAX_BOOKS_PROFILE:] if len(user_books) > MAX_BOOKS_PROFILE else user_books
        )

        indices = [book_id_to_idx[asin] for asin in recent_books if asin in book_id_to_idx]

        if not indices:
            skipped_users.append(uid)
            continue

        user_vectors = book_embeddings[indices]

        mean_vector = np.mean(
            user_vectors,
            axis=0,
        ).astype(np.float32)

        mean_vector = mean_vector.reshape(1, -1)

        faiss.normalize_L2(mean_vector)

        mean_vector = mean_vector[0]

        h2_profiles.append(mean_vector)

        valid_users.append(uid)

    # ─────────────────────────────────────────────────────────────────────
    # Final Array
    # ─────────────────────────────────────────────────────────────────────

    h2_array = np.ascontiguousarray(np.array(h2_profiles, dtype=np.float32))

    # ─────────────────────────────────────────────────────────────────────
    # Save Artifacts
    # ─────────────────────────────────────────────────────────────────────

    domain_tag = f"_{domain}" if domain != "music" else ""
    npy_path = PROCESSED_DIR / f"user_profiles_h2{domain_tag}.npy"

    idx_path = PROCESSED_DIR / f"user_profile_index_h2{domain_tag}.json"

    metadata_path = PROCESSED_DIR / f"user_profiles_h2{domain_tag}_metadata.json"

    print("\nSaving profile matrix...")

    np.save(
        npy_path,
        h2_array,
    )

    print("Saving index mapping...")

    index_map = {uid: idx for idx, uid in enumerate(valid_users)}

    save_json(
        idx_path,
        index_map,
    )

    print("Saving experiment metadata...")

    metadata = {
        "heuristic": "H2",
        "description": ("Mean pooled book embedding baseline for cross-domain recommendation"),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": embedding_dim,
        "normalized_embeddings": True,
        "aggregation_strategy": "mean_pooling",
        "max_books_profile": MAX_BOOKS_PROFILE,
        "evaluation_users_requested": N_TEST_USERS,
        "evaluation_users_generated": len(valid_users),
        "skipped_users": len(skipped_users),
        "seed": SEED,
        "generated_at": datetime.now(UTC).isoformat(),
        "artifacts": {
            "profiles": npy_path.name,
            "index_map": idx_path.name,
        },
    }

    save_json(
        metadata_path,
        metadata,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Final Report
    # ─────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("H2 GENERATION COMPLETE")
    print("=" * 80)

    print(f"Embedding model       : {EMBEDDING_MODEL}")
    print(f"Embedding dimension   : {embedding_dim}")
    print(f"Profile window        : {MAX_BOOKS_PROFILE}")

    print(f"\nUsers requested       : {N_TEST_USERS:,}")
    print(f"Users generated       : {len(valid_users):,}")
    print(f"Users skipped         : {len(skipped_users):,}")

    print(f"\nFinal tensor shape    : {h2_array.shape}")

    print("\nSaved artifacts:")
    print(f"  • {npy_path}")
    print(f"  • {idx_path}")
    print(f"  • {metadata_path}")

    print("=" * 80)
