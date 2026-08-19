"""Frozen-space nearest-neighbour clinical-sharing analysis for Step 7."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


PRIMARY_K_NEIGHBORS = 10
SENSITIVITY_K_NEIGHBORS = (5, 20)
DISTANCE_METRIC = "euclidean"
N_BOOTSTRAP = 2000
N_RANDOM_BASELINE_REPEATS = 2000
N_PERMUTATIONS = 2000
RANDOM_SEED = 42
SITE_MINIMUM_CANDIDATES = PRIMARY_K_NEIGHBORS
SAMPLE_WITH_REPLACEMENT = False

HOUSE_NAVY = "#003366"
HOUSE_CRIMSON = "#BA2828"
HOUSE_TEAL = "#5BBABA"
HOUSE_KEY_RED = "#FF0000"
HOUSE_GRAY = "#888888"
POSITIVE_COLOR = HOUSE_CRIMSON
NEGATIVE_COLOR = HOUSE_NAVY
CORRECTED_COLOR = HOUSE_TEAL
NULL_COLOR = HOUSE_GRAY
KEY_EVENT_COLOR = HOUSE_KEY_RED

REPRESENTATION_DIMENSIONS = tuple(f"r_{index:03d}" for index in range(128))
CONDITIONS = ("full_all", "neutral_all")
CONDITION_LABELS = {
    "full_all": "Full profile",
    "neutral_all": "Static neutral",
}
CONDITION_MARKERS = {
    "full_all": "s",
    "neutral_all": "o",
}
VARIABLES = (
    ("mean_glucose", "Mean glucose", "continuous"),
    ("glucose_cv", "Glucose CV", "continuous"),
    ("tir_70_180", "Time in range", "continuous"),
    ("glucose_sd", "Glucose SD", "continuous"),
    ("hba1c", "HbA1c", "continuous"),
    ("study_group", "Study group", "categorical"),
    (
        "natriuretic_peptide_b_prohormon",
        "NT-proBNP",
        "continuous",
    ),
    ("c_reactive_protein_i", "High-sensitivity CRP", "continuous"),
    ("bun_creatinine_ratio", "BUN/creatinine ratio", "continuous"),
)
SKEWED_BIOMARKERS = {
    "natriuretic_peptide_b_prohormon",
    "c_reactive_protein_i",
    "bun_creatinine_ratio",
}
NO_EM_DASH = "\u2014"


def stable_seed(*parts: Any, base_seed: int = RANDOM_SEED) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(
        hashlib.sha256(payload).digest()[:8], byteorder="little"
    )
    return int((value + base_seed) % (2**32 - 1))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    def converter(item: Any) -> Any:
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating,)):
            return None if not np.isfinite(item) else float(item)
        if isinstance(item, (np.bool_,)):
            return bool(item)
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=converter) + "\n"
    )
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    output = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return output
    selected = values[finite]
    order = np.argsort(selected)
    ranked = selected[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    reordered = np.empty_like(adjusted)
    reordered[order] = adjusted
    output[finite] = reordered
    return output


def load_frozen_scores(
    representations_path: Path,
    pipeline_root: Path,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    frame = pd.read_parquet(representations_path)
    frame["participant_id"] = frame["participant_id"].astype(str)
    participant_sets: dict[str, set[str]] = {}
    condition_frames: dict[str, pd.DataFrame] = {}
    for condition in CONDITIONS:
        selected = frame.loc[
            frame["representation_type"] == condition
        ].copy()
        if selected.duplicated("participant_id").any():
            raise RuntimeError(f"Duplicate {condition} participant rows")
        if set(selected["split"].astype(str)) != {"test"}:
            raise RuntimeError(f"Non-test row entered {condition}")
        participant_sets[condition] = set(selected["participant_id"])
        condition_frames[condition] = selected.set_index("participant_id")
    if participant_sets["full_all"] != participant_sets["neutral_all"]:
        raise RuntimeError("Full and neutral participant sets differ")
    participant_ids = sorted(participant_sets["full_all"])
    if len(participant_ids) != 221:
        raise RuntimeError(
            f"Expected 221 test participants, found {len(participant_ids)}"
        )
    scores: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        condition_dir = pipeline_root / condition
        feature_order_path = condition_dir / "feature_order.json"
        kept_path = condition_dir / "kept_dimensions.npy"
        scaler_path = condition_dir / f"{condition}_scaler.joblib"
        pca_path = condition_dir / f"{condition}_pca.joblib"
        required = (
            feature_order_path,
            kept_path,
            scaler_path,
            pca_path,
        )
        if any(not path.exists() for path in required):
            raise RuntimeError(
                f"Incomplete frozen pipeline for {condition}: {required}"
            )
        feature_order = json.loads(feature_order_path.read_text())
        source_dimensions = feature_order["source_dimensions"]
        if source_dimensions != list(REPRESENTATION_DIMENSIONS):
            raise RuntimeError(f"Feature order mismatch for {condition}")
        kept = np.load(kept_path)
        if not np.array_equal(
            kept, np.asarray(feature_order["kept_indices"], dtype=int)
        ):
            raise RuntimeError(f"Kept-dimension mismatch for {condition}")
        raw = condition_frames[condition].reindex(participant_ids).loc[
            :, source_dimensions
        ].to_numpy(float)
        if not np.isfinite(raw).all():
            raise RuntimeError(f"Nonfinite representation for {condition}")
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)
        retained = int(feature_order["primary_components"])
        if scaler.n_features_in_ != len(kept):
            raise RuntimeError(f"Scaler dimension mismatch for {condition}")
        if pca.n_features_in_ != len(kept):
            raise RuntimeError(f"PCA dimension mismatch for {condition}")
        transformed = pca.transform(scaler.transform(raw[:, kept]))
        scores[condition] = transformed[:, :retained]
        if not np.isfinite(scores[condition]).all():
            raise RuntimeError(f"Nonfinite frozen scores for {condition}")
        metadata[condition] = {
            "participant_count": len(participant_ids),
            "raw_dimensions": raw.shape[1],
            "retained_components": retained,
            "explained_variance_ratio": (
                pca.explained_variance_ratio_[:retained].tolist()
            ),
            "explained_variance_sum": float(
                pca.explained_variance_ratio_[:retained].sum()
            ),
            "feature_order_path": str(feature_order_path),
            "kept_dimensions_path": str(kept_path),
            "scaler_path": str(scaler_path),
            "pca_path": str(pca_path),
            "feature_order_hash": sha256_file(feature_order_path),
            "kept_dimensions_hash": sha256_file(kept_path),
            "scaler_hash": sha256_file(scaler_path),
            "pca_hash": sha256_file(pca_path),
            "fitted_on_test": False,
        }
    return participant_ids, scores, metadata


def load_clinical_frame(
    participant_ids: list[str],
    test_features_path: Path,
) -> pd.DataFrame:
    frame = pd.read_parquet(test_features_path)
    frame["participant_id"] = frame["participant_id"].astype(str)
    if frame.duplicated("participant_id").any():
        raise RuntimeError("Duplicate participant in test clinical features")
    if set(frame["participant_id"]) != set(participant_ids):
        raise RuntimeError("Test clinical feature participants do not match")
    required = {
        "clinical_site",
        "study_group",
        *[variable for variable, _, _ in VARIABLES],
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing Tier 1 clinical columns: {missing}")
    return frame.set_index("participant_id").reindex(participant_ids).reset_index()


def pairwise_euclidean(values: np.ndarray) -> np.ndarray:
    squared_norm = np.sum(values * values, axis=1)
    squared = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * values.dot(values.T)
    )
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared)


def build_graphs(
    participant_ids: list[str],
    scores: dict[str, np.ndarray],
    clinical: pd.DataFrame,
    k_values: tuple[int, ...],
) -> tuple[dict[tuple[str, int], np.ndarray], pd.DataFrame]:
    graphs: dict[tuple[str, int], np.ndarray] = {}
    edge_rows: list[dict[str, Any]] = []
    sites = clinical["clinical_site"].astype(str).to_numpy()
    groups = clinical["study_group"].astype(str).to_numpy()
    for condition in CONDITIONS:
        distances = pairwise_euclidean(scores[condition])
        np.fill_diagonal(distances, np.inf)
        ordering = np.argsort(distances, axis=1)
        for k_neighbors in k_values:
            neighbors = ordering[:, :k_neighbors]
            graphs[(condition, k_neighbors)] = neighbors
            for focal_index, neighbor_indices in enumerate(neighbors):
                for rank, neighbor_index in enumerate(neighbor_indices, start=1):
                    edge_rows.append(
                        {
                            "condition": condition,
                            "k_neighbors": k_neighbors,
                            "focal_participant_id":
                                participant_ids[focal_index],
                            "neighbor_participant_id":
                                participant_ids[neighbor_index],
                            "neighbor_rank": rank,
                            "distance": distances[
                                focal_index, neighbor_index
                            ],
                            "focal_site": sites[focal_index],
                            "neighbor_site": sites[neighbor_index],
                            "same_site":
                                sites[focal_index] == sites[neighbor_index],
                            "focal_study_group": groups[focal_index],
                            "neighbor_study_group":
                                groups[neighbor_index],
                            "same_study_group":
                                groups[focal_index] == groups[neighbor_index],
                        }
                    )
    return graphs, pd.DataFrame(edge_rows)


def cyclic_sample_indices(
    candidate_indices: np.ndarray,
    sample_size: int,
    repeats: int,
    seed: int,
) -> np.ndarray:
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    candidate_count = len(candidate_indices)
    sample_size = min(sample_size, candidate_count)
    if sample_size <= 0:
        return np.empty((repeats, 0), dtype=int)
    if sample_size == candidate_count:
        return np.broadcast_to(
            candidate_indices[None, :], (repeats, candidate_count)
        ).copy()
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, candidate_count, size=repeats)
    coprime_steps = np.asarray(
        [
            step
            for step in range(1, candidate_count)
            if math.gcd(step, candidate_count) == 1
        ],
        dtype=int,
    )
    if not len(coprime_steps):
        coprime_steps = np.asarray([1], dtype=int)
    steps = rng.choice(coprime_steps, size=repeats, replace=True)
    offsets = np.arange(sample_size, dtype=int)
    positions = (
        starts[:, None] + steps[:, None] * offsets[None, :]
    ) % candidate_count
    sampled = candidate_indices[positions]
    if sample_size > 1:
        sorted_rows = np.sort(sampled, axis=1)
        if np.any(np.diff(sorted_rows, axis=1) == 0):
            raise RuntimeError("Random comparison set sampled with replacement")
    return sampled


def bootstrap_mean_ci(
    values: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(
        0, len(values), size=(repeats, len(values))
    )
    means = values[draw_indices].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]).tolist())


def rank_values(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    ranked = np.full(len(values), np.nan)
    ranked[observed] = (
        pd.Series(values[observed]).rank(method="average").to_numpy()
        / observed.sum()
    )
    return ranked


def participant_metrics_and_weights(
    values: np.ndarray,
    variable_type: str,
    sites: np.ndarray,
    neighbors: np.ndarray,
    condition: str,
    variable: str,
    k_neighbors: int,
    random_repeats: int,
    seed: int,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, int]:
    participant_count = len(values)
    if variable_type == "continuous":
        numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()
        observed = np.isfinite(numeric)
        observed_values = numeric
        observed_sd = np.std(numeric[observed], ddof=1)
        standardized = np.full(participant_count, np.nan)
        standardized[observed] = (
            numeric[observed] - np.mean(numeric[observed])
        ) / observed_sd
        ranked = (
            rank_values(numeric, observed)
            if variable in SKEWED_BIOMARKERS
            else np.full(participant_count, np.nan)
        )
    else:
        text = pd.Series(values, dtype="object")
        observed = text.notna().to_numpy()
        observed_values = text.astype(str).to_numpy()
        standardized = np.full(participant_count, np.nan)
        ranked = np.full(participant_count, np.nan)
    unrestricted_weights = np.zeros((participant_count, participant_count))
    site_weights = np.zeros((participant_count, participant_count))
    rows: list[dict[str, Any]] = []
    insufficient_site = 0
    all_indices = np.arange(participant_count)
    for focal_index in all_indices[observed]:
        selected_neighbors = neighbors[focal_index]
        valid_neighbors = selected_neighbors[observed[selected_neighbors]]
        if not len(valid_neighbors):
            continue
        candidate_mask = observed.copy()
        candidate_mask[focal_index] = False
        candidate_mask[selected_neighbors] = False
        unrestricted_candidates = all_indices[candidate_mask]
        random_indices = cyclic_sample_indices(
            unrestricted_candidates,
            len(valid_neighbors),
            random_repeats,
            stable_seed(
                condition,
                variable,
                k_neighbors,
                focal_index,
                "unrestricted",
                base_seed=seed,
            ),
        )
        if not random_indices.shape[1]:
            continue
        site_candidate_mask = candidate_mask & (sites == sites[focal_index])
        site_candidates = all_indices[site_candidate_mask]
        if len(site_candidates) < k_neighbors:
            insufficient_site += 1
        site_random_indices = cyclic_sample_indices(
            site_candidates,
            len(valid_neighbors),
            random_repeats,
            stable_seed(
                condition,
                variable,
                k_neighbors,
                focal_index,
                "site",
                base_seed=seed,
            ),
        )
        neighbor_weights = np.bincount(
            valid_neighbors, minlength=participant_count
        ).astype(float)
        neighbor_weights /= neighbor_weights.sum()
        random_weights = np.bincount(
            random_indices.ravel(), minlength=participant_count
        ).astype(float)
        random_weights /= random_weights.sum()
        unrestricted_weights[focal_index] = (
            random_weights - neighbor_weights
            if variable_type == "continuous"
            else neighbor_weights - random_weights
        )
        if site_random_indices.shape[1]:
            site_random_weights = np.bincount(
                site_random_indices.ravel(), minlength=participant_count
            ).astype(float)
            site_random_weights /= site_random_weights.sum()
            site_weights[focal_index] = (
                site_random_weights - neighbor_weights
                if variable_type == "continuous"
                else neighbor_weights - site_random_weights
            )
        if variable_type == "continuous":
            neighbor_raw = np.mean(
                np.abs(
                    observed_values[focal_index]
                    - observed_values[valid_neighbors]
                )
            )
            random_raw = np.mean(
                np.abs(
                    observed_values[focal_index]
                    - observed_values[random_indices]
                )
            )
            neighbor_standardized = np.mean(
                np.abs(
                    standardized[focal_index]
                    - standardized[valid_neighbors]
                )
            )
            random_standardized = np.mean(
                np.abs(
                    standardized[focal_index]
                    - standardized[random_indices]
                )
            )
            site_random_raw = (
                np.mean(
                    np.abs(
                        observed_values[focal_index]
                        - observed_values[site_random_indices]
                    )
                )
                if site_random_indices.shape[1]
                else np.nan
            )
            site_random_standardized = (
                np.mean(
                    np.abs(
                        standardized[focal_index]
                        - standardized[site_random_indices]
                    )
                )
                if site_random_indices.shape[1]
                else np.nan
            )
            rank_neighbor = (
                np.mean(
                    np.abs(ranked[focal_index] - ranked[valid_neighbors])
                )
                if variable in SKEWED_BIOMARKERS
                else np.nan
            )
            rank_random = (
                np.mean(
                    np.abs(ranked[focal_index] - ranked[random_indices])
                )
                if variable in SKEWED_BIOMARKERS
                else np.nan
            )
            rank_site_random = (
                np.mean(
                    np.abs(
                        ranked[focal_index] - ranked[site_random_indices]
                    )
                )
                if variable in SKEWED_BIOMARKERS
                and site_random_indices.shape[1]
                else np.nan
            )
            neighbor_metric = neighbor_standardized
            random_metric = random_standardized
            sharing_gain = random_standardized - neighbor_standardized
            site_random_metric = site_random_standardized
            site_gain = (
                site_random_standardized - neighbor_standardized
                if np.isfinite(site_random_standardized)
                else np.nan
            )
            same_neighbor = np.nan
            same_random = np.nan
            same_site_random = np.nan
        else:
            same_neighbor = np.mean(
                observed_values[valid_neighbors]
                == observed_values[focal_index]
            )
            same_random = np.mean(
                observed_values[random_indices]
                == observed_values[focal_index]
            )
            same_site_random = (
                np.mean(
                    observed_values[site_random_indices]
                    == observed_values[focal_index]
                )
                if site_random_indices.shape[1]
                else np.nan
            )
            neighbor_raw = np.nan
            random_raw = np.nan
            neighbor_standardized = np.nan
            random_standardized = np.nan
            site_random_raw = np.nan
            site_random_standardized = np.nan
            rank_neighbor = np.nan
            rank_random = np.nan
            rank_site_random = np.nan
            neighbor_metric = same_neighbor
            random_metric = same_random
            sharing_gain = same_neighbor - same_random
            site_random_metric = same_site_random
            site_gain = (
                same_neighbor - same_site_random
                if np.isfinite(same_site_random)
                else np.nan
            )
        rows.append(
            {
                "focal_index": focal_index,
                "neighbor_metric": neighbor_metric,
                "random_metric": random_metric,
                "sharing_gain": sharing_gain,
                "site_matched_random_metric": site_random_metric,
                "site_matched_sharing_gain": site_gain,
                "neighbor_raw_metric": neighbor_raw,
                "random_raw_metric": random_raw,
                "site_matched_random_raw_metric": site_random_raw,
                "neighbor_standardized_metric": neighbor_standardized,
                "random_standardized_metric": random_standardized,
                "site_matched_random_standardized_metric":
                    site_random_standardized,
                "same_group_rate_neighbors": same_neighbor,
                "same_group_rate_random": same_random,
                "same_group_rate_site_matched_random": same_site_random,
                "rank_neighbor_metric": rank_neighbor,
                "rank_random_metric": rank_random,
                "rank_site_matched_random_metric": rank_site_random,
                "rank_sharing_gain":
                    rank_random - rank_neighbor
                    if np.isfinite(rank_random)
                    else np.nan,
                "rank_site_matched_sharing_gain":
                    rank_site_random - rank_neighbor
                    if np.isfinite(rank_site_random)
                    else np.nan,
                "n_valid_neighbors": len(valid_neighbors),
                "n_valid_random_candidates": len(unrestricted_candidates),
                "n_valid_site_candidates": len(site_candidates),
                "site_candidate_insufficient":
                    len(site_candidates) < k_neighbors,
            }
        )
    return rows, unrestricted_weights, site_weights, insufficient_site


def permuted_values(
    values: np.ndarray,
    observed: np.ndarray,
    sites: np.ndarray,
    repeats: int,
    seed: int,
    within_site: bool,
) -> np.ndarray:
    eligible_indices = np.flatnonzero(observed)
    base = np.asarray(values)[eligible_indices]
    rng = np.random.default_rng(seed)
    permutations = np.empty((repeats, len(eligible_indices)), dtype=base.dtype)
    if within_site:
        eligible_sites = sites[eligible_indices]
        site_positions = [
            np.flatnonzero(eligible_sites == site)
            for site in sorted(set(eligible_sites))
        ]
        if any(len(positions) < 2 for positions in site_positions):
            within_site = False
    for repeat in range(repeats):
        if within_site:
            row = base.copy()
            for positions in site_positions:
                row[positions] = base[positions][rng.permutation(len(positions))]
            permutations[repeat] = row
        else:
            permutations[repeat] = base[rng.permutation(len(base))]
    return permutations


def weighted_permutation_metrics(
    permutation_rows: np.ndarray,
    observed_indices: np.ndarray,
    weights: np.ndarray,
    variable_type: str,
    batch_size: int = 100,
) -> np.ndarray:
    reduced_weights = weights[np.ix_(observed_indices, observed_indices)]
    combined = reduced_weights + reduced_weights.T
    upper_i, upper_j = np.triu_indices(len(observed_indices), k=1)
    edge_weights = combined[upper_i, upper_j]
    keep = np.abs(edge_weights) > 1e-15
    upper_i = upper_i[keep]
    upper_j = upper_j[keep]
    edge_weights = edge_weights[keep]
    output = np.empty(len(permutation_rows), dtype=float)
    for start in range(0, len(permutation_rows), batch_size):
        stop = min(start + batch_size, len(permutation_rows))
        left = permutation_rows[start:stop, upper_i]
        right = permutation_rows[start:stop, upper_j]
        pair_metric = (
            np.abs(left - right)
            if variable_type == "continuous"
            else (left == right).astype(float)
        )
        output[start:stop] = pair_metric.dot(edge_weights)
    return output


def empirical_two_sided_p(null_values: np.ndarray, estimate: float) -> float:
    return float(
        (1 + np.sum(np.abs(null_values) >= abs(estimate)))
        / (len(null_values) + 1)
    )


def summarize_result(
    participant_frame: pd.DataFrame,
    variable: str,
    variable_label: str,
    variable_type: str,
    condition: str,
    k_neighbors: int,
    baseline_type: str,
    insufficient_site: int,
    weights: np.ndarray,
    permutation_rows: np.ndarray,
    observed_indices: np.ndarray,
    bootstrap_repeats: int,
    permutation_repeats: int,
    seed: int,
    permutation_scheme: str,
) -> dict[str, Any]:
    site_matched = baseline_type == "site_matched"
    gain_column = (
        "site_matched_sharing_gain" if site_matched else "sharing_gain"
    )
    random_column = (
        "site_matched_random_metric" if site_matched else "random_metric"
    )
    random_raw_column = (
        "site_matched_random_raw_metric"
        if site_matched
        else "random_raw_metric"
    )
    random_standardized_column = (
        "site_matched_random_standardized_metric"
        if site_matched
        else "random_standardized_metric"
    )
    rank_random_column = (
        "rank_site_matched_random_metric"
        if site_matched
        else "rank_random_metric"
    )
    rank_gain_column = (
        "rank_site_matched_sharing_gain"
        if site_matched
        else "rank_sharing_gain"
    )
    eligible = participant_frame.loc[
        np.isfinite(participant_frame[gain_column])
    ].copy()
    estimate = float(eligible[gain_column].mean())
    ci_low, ci_high = bootstrap_mean_ci(
        eligible[gain_column].to_numpy(float),
        bootstrap_repeats,
        stable_seed(
            variable,
            condition,
            k_neighbors,
            baseline_type,
            "bootstrap",
            base_seed=seed,
        ),
    )
    null_values = weighted_permutation_metrics(
        permutation_rows,
        observed_indices,
        weights / len(eligible),
        variable_type,
    )
    permutation_p = empirical_two_sided_p(null_values, estimate)
    rank_ci_low = np.nan
    rank_ci_high = np.nan
    if variable in SKEWED_BIOMARKERS:
        rank_ci_low, rank_ci_high = bootstrap_mean_ci(
            eligible[rank_gain_column].to_numpy(float),
            bootstrap_repeats,
            stable_seed(
                variable,
                condition,
                k_neighbors,
                baseline_type,
                "rank_bootstrap",
                base_seed=seed,
            ),
        )
    if not site_matched and k_neighbors == PRIMARY_K_NEIGHBORS:
        fdr_family = (
            "primary_neutral_k10_tier1"
            if condition == "neutral_all"
            else "secondary_full_k10_tier1"
        )
    elif not site_matched:
        fdr_family = f"sensitivity_{condition}_k{k_neighbors}_tier1"
    else:
        fdr_family = "site_matched_sensitivity"
    if variable_type == "continuous":
        neighbor_raw = float(eligible["neighbor_raw_metric"].mean())
        random_raw = float(eligible[random_raw_column].mean())
        neighbor_standardized = float(
            eligible["neighbor_standardized_metric"].mean()
        )
        random_standardized = float(
            eligible[random_standardized_column].mean()
        )
        same_neighbor = np.nan
        same_random = np.nan
        same_gain = np.nan
    else:
        neighbor_raw = np.nan
        random_raw = np.nan
        neighbor_standardized = np.nan
        random_standardized = np.nan
        same_neighbor = float(eligible["neighbor_metric"].mean())
        same_random = float(eligible[random_column].mean())
        same_gain = estimate
    return {
        "variable": variable,
        "variable_label": variable_label,
        "variable_type": variable_type,
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "k_neighbors": k_neighbors,
        "distance_metric": DISTANCE_METRIC,
        "n_participants_eligible": len(eligible),
        "n_directed_neighbor_pairs": int(
            eligible["n_valid_neighbors"].sum()
        ),
        "neighbor_raw_mean_difference": neighbor_raw,
        "random_raw_mean_difference": random_raw,
        "raw_neighbor_minus_random": neighbor_raw - random_raw
        if variable_type == "continuous"
        else np.nan,
        "neighbor_standardized_difference": neighbor_standardized,
        "random_standardized_difference": random_standardized,
        "standardized_similarity_gain": estimate
        if variable_type == "continuous"
        else same_gain,
        "same_group_rate_neighbors": same_neighbor,
        "same_group_rate_random": same_random,
        "same_group_rate_gain": same_gain,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "permutation_p": permutation_p,
        "fdr_family": fdr_family,
        "fdr_q": np.nan,
        "random_baseline_type": baseline_type,
        "site_matched": site_matched,
        "n_insufficient_site_candidates": insufficient_site
        if site_matched
        else 0,
        "rank_neighbor_difference": float(
            eligible["rank_neighbor_metric"].mean()
        )
        if variable in SKEWED_BIOMARKERS
        else np.nan,
        "rank_random_difference": float(
            eligible[rank_random_column].mean()
        )
        if variable in SKEWED_BIOMARKERS
        else np.nan,
        "rank_similarity_gain": float(eligible[rank_gain_column].mean())
        if variable in SKEWED_BIOMARKERS
        else np.nan,
        "rank_bootstrap_ci_low": rank_ci_low,
        "rank_bootstrap_ci_high": rank_ci_high,
        "permutation_scheme": permutation_scheme,
        "random_baseline_repeats": N_RANDOM_BASELINE_REPEATS,
        "bootstrap_replicates": bootstrap_repeats,
        "permutation_replicates": permutation_repeats,
        "analysis_status": "complete",
        "notes": (
            "Full profile includes real participant-specific static conditioning."
            if condition == "full_all"
            else "Participant-specific static conditioning is neutralized."
        ),
    }


def add_fdr(results: pd.DataFrame) -> pd.DataFrame:
    adjusted = results.copy()
    families = [
        "primary_neutral_k10_tier1",
        "secondary_full_k10_tier1",
        "sensitivity_full_all_k5_tier1",
        "sensitivity_full_all_k20_tier1",
        "sensitivity_neutral_all_k5_tier1",
        "sensitivity_neutral_all_k20_tier1",
    ]
    for family in families:
        mask = adjusted["fdr_family"] == family
        expected_count = 9
        if mask.sum() != expected_count:
            raise RuntimeError(
                f"FDR family {family} has {mask.sum()} rows, expected 9"
            )
        adjusted.loc[mask, "fdr_q"] = bh_adjust(
            adjusted.loc[mask, "permutation_p"].to_numpy(float)
        )
    primary = adjusted[
        adjusted["fdr_family"] == "primary_neutral_k10_tier1"
    ]
    secondary = adjusted[
        adjusted["fdr_family"] == "secondary_full_k10_tier1"
    ]
    if len(primary) != 9 or len(secondary) != 9:
        raise RuntimeError("Frozen primary or secondary FDR family changed")
    return adjusted


def estimate_color(row: pd.Series) -> str:
    if row["bootstrap_ci_low"] > 0:
        return POSITIVE_COLOR
    if row["bootstrap_ci_high"] < 0:
        return NEGATIVE_COLOR
    return NULL_COLOR


def make_forest_figure(results: pd.DataFrame, output_path: Path) -> None:
    primary = results[
        (results["k_neighbors"] == PRIMARY_K_NEIGHBORS)
        & (~results["site_matched"])
    ].copy()
    variable_order = [label for _, label, _ in VARIABLES]
    y_positions = {
        label: len(variable_order) - 1 - index
        for index, label in enumerate(variable_order)
    }
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(11.5, 7.3))
    offsets = {"full_all": -0.12, "neutral_all": 0.12}
    for condition in CONDITIONS:
        condition_rows = primary[primary["condition"] == condition]
        for _, row in condition_rows.iterrows():
            y_value = y_positions[row["variable_label"]] + offsets[condition]
            estimate = row["standardized_similarity_gain"]
            lower = estimate - row["bootstrap_ci_low"]
            upper = row["bootstrap_ci_high"] - estimate
            axis.errorbar(
                estimate,
                y_value,
                xerr=np.asarray([[lower], [upper]]),
                fmt=CONDITION_MARKERS[condition],
                markersize=7,
                markerfacecolor=estimate_color(row),
                markeredgecolor=estimate_color(row),
                ecolor=estimate_color(row),
                elinewidth=1.7,
                capsize=3,
                zorder=4 if condition == "neutral_all" else 3,
            )
            if condition == "neutral_all":
                axis.annotate(
                    f"q={row['fdr_q']:.3f}",
                    (row["bootstrap_ci_high"], y_value),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color=HOUSE_GRAY,
                )
    axis.axvline(0.0, color=HOUSE_GRAY, linewidth=1.2, linestyle="--")
    axis.set_yticks(
        [y_positions[label] for label in variable_order],
        labels=variable_order,
    )
    axis.set_xlabel(
        "Clinical similarity gain versus random non-neighbours\n"
        "Positive values indicate stronger sharing among hidden-state neighbours"
    )
    axis.set_title(
        "Clinical sharing among nearest hidden-state neighbours\n"
        "Test participants, k=10, 95% participant-bootstrap confidence intervals",
        loc="left",
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=CONDITION_MARKERS["full_all"],
            color=HOUSE_GRAY,
            markerfacecolor=HOUSE_GRAY,
            linestyle="None",
            label="Full profile",
        ),
        Line2D(
            [0],
            [0],
            marker=CONDITION_MARKERS["neutral_all"],
            color=HOUSE_GRAY,
            markerfacecolor=HOUSE_GRAY,
            linestyle="None",
            label="Static neutral",
        ),
    ]
    axis.legend(
        handles=legend_handles,
        title="Representation space",
        loc="lower right",
    )
    axis.text(
        0.01,
        -0.15,
        "Full-profile results include real participant-specific static conditioning. "
        "Colour indicates whether the confidence interval is above, below, or "
        "crosses zero.",
        transform=axis.transAxes,
        fontsize=8.5,
        color=HOUSE_GRAY,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def supported(row: pd.Series) -> bool:
    return bool(
        row["bootstrap_ci_low"] > 0
        and row["permutation_p"] < 0.05
    )


def format_result(row: pd.Series, include_q: bool = False) -> str:
    text = (
        f"{row['variable_label']}: {row['standardized_similarity_gain']:.3f} "
        f"[{row['bootstrap_ci_low']:.3f}, "
        f"{row['bootstrap_ci_high']:.3f}], "
        f"p={row['permutation_p']:.4f}"
    )
    if include_q:
        text += f", q={row['fdr_q']:.4f}"
    return text


def make_report(results: pd.DataFrame, output_path: Path) -> str:
    primary = results[
        (results["condition"] == "neutral_all")
        & (results["k_neighbors"] == PRIMARY_K_NEIGHBORS)
        & (~results["site_matched"])
    ].set_index("variable")
    full = results[
        (results["condition"] == "full_all")
        & (results["k_neighbors"] == PRIMARY_K_NEIGHBORS)
        & (~results["site_matched"])
    ].set_index("variable")
    supported_neutral = [
        primary.loc[variable, "variable_label"]
        for variable, _, _ in VARIABLES
        if supported(primary.loc[variable])
        and primary.loc[variable, "fdr_q"] < 0.05
    ]
    external_supported = [
        primary.loc[variable, "variable_label"]
        for variable in SKEWED_BIOMARKERS
        if supported(primary.loc[variable])
        and primary.loc[variable, "fdr_q"] < 0.05
    ]
    lines = [
        "# Nearest-neighbour clinical-sharing report",
        "",
        "## 1. Objective",
        "",
        "Test whether participants nearest in frozen hidden-state space were "
        "more clinically similar than randomly selected non-neighbours.",
        "",
        "## 2. Why neighbour sharing is preferable to forced clustering",
        "",
        "The prior analysis supported a continuous glycemic manifold but no "
        "primary discrete solution. Nearest-neighbour analysis tests local "
        "clinical sharing directly without imposing cluster boundaries.",
        "",
        "## 3. Frozen representation spaces",
        "",
        "The analysis used the full-profile and static-neutral test "
        "representations transformed by their matching serialized validation "
        "scalers and PCA models. Nothing was fitted on test.",
        "",
        "## 4. Distance and neighbour definition",
        "",
        "Euclidean distance was calculated in each retained frozen PCA space. "
        "The primary graph used k=10 directed neighbours after excluding self. "
        "Inference operated at the focal-participant level.",
        "",
        "## 5. Random baselines",
        "",
        "The primary baseline used 2,000 deterministic, without-replacement "
        "random non-neighbour sets from the eligible test cohort. The "
        "sensitivity baseline was restricted to the focal participant's "
        "clinical site.",
        "",
        "## 6. Positive-control glucose sharing",
        "",
    ]
    for variable in (
        "mean_glucose",
        "glucose_cv",
        "tir_70_180",
        "glucose_sd",
    ):
        lines.append("- " + format_result(primary.loc[variable], include_q=True))
    lines.extend(
        [
            "",
            "## 7. HbA1c sharing",
            "",
            format_result(primary.loc["hba1c"], include_q=True),
            "",
            "## 8. Study-group sharing",
            "",
            format_result(primary.loc["study_group"], include_q=True),
            "",
            "## 9. External biomarker sharing",
            "",
        ]
    )
    for variable in (
        "natriuretic_peptide_b_prohormon",
        "c_reactive_protein_i",
        "bun_creatinine_ratio",
    ):
        lines.append("- " + format_result(primary.loc[variable], include_q=True))
    lines.extend(
        [
            "",
            "External variables meeting both participant-bootstrap, "
            "permutation, and primary FDR criteria: "
            + (", ".join(external_supported) if external_supported else "none")
            + ".",
            "",
            "## 10. Full-versus-neutral comparison",
            "",
        ]
    )
    comparison = pd.DataFrame(
        {
            "Variable": [label for _, label, _ in VARIABLES],
            "Full profile": [
                full.loc[variable, "standardized_similarity_gain"]
                for variable, _, _ in VARIABLES
            ],
            "Static neutral": [
                primary.loc[variable, "standardized_similarity_gain"]
                for variable, _, _ in VARIABLES
            ],
        }
    )
    lines.append("```csv\n" + comparison.to_csv(index=False, float_format="%.3f").strip() + "\n```")
    site = results[
        (results["condition"] == "neutral_all")
        & (results["k_neighbors"] == PRIMARY_K_NEIGHBORS)
        & (results["site_matched"])
    ]
    lines.extend(
        [
            "",
            "## 11. Site-matched sensitivity",
            "",
            "```csv\n" + site[[
                "variable_label",
                "standardized_similarity_gain",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "permutation_p",
                "n_insufficient_site_candidates",
            ]].to_csv(index=False, float_format="%.3f").strip() + "\n```",
            "",
            "## 12. k=5 and k=20 sensitivity",
            "",
        ]
    )
    sensitivity = results[
        (results["condition"] == "neutral_all")
        & (results["k_neighbors"].isin(SENSITIVITY_K_NEIGHBORS))
        & (~results["site_matched"])
    ]
    lines.append(
        "```csv\n" + sensitivity[[
            "k_neighbors",
            "variable_label",
            "standardized_similarity_gain",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "permutation_p",
        ]].to_csv(index=False, float_format="%.3f").strip() + "\n```"
    )
    lines.extend(
        [
            "",
            "## 13. Limitations",
            "",
            "Neighbour graphs are directed and local. Variables with missing "
            "measurements contribute fewer valid directed pairs. The random "
            "baseline is conditional on the observed test cohort. Full-profile "
            "sharing includes participant-specific static conditioning and is "
            "not a causal decomposition.",
            "",
            "## 14. Preliminary answer to the original question",
            "",
            "The static-neutral test manifold showed FDR-supported local "
            "clinical sharing for: "
            + (", ".join(supported_neutral) if supported_neutral else "none")
            + ". External biomarker sharing was "
            + (
                "supported for " + ", ".join(external_supported)
                if external_supported
                else "not supported after primary multiplicity correction"
            )
            + ". This is the preliminary Gate 2 interpretation and does not "
            "yet include the targeted HbA1c predictive closing analysis.",
        ]
    )
    text = "\n".join(lines) + "\n"
    output_path.write_text(text)
    return text


def scan_em_dash(paths: list[Path]) -> list[str]:
    affected: list[str] = []
    for root in paths:
        candidates = root.rglob("*") if root.is_dir() else [root]
        for path in candidates:
            if path.is_file() and path.suffix.lower() in {
                ".csv", ".json", ".log", ".md", ".py", ".tex"
            }:
                try:
                    if NO_EM_DASH in path.read_text():
                        affected.append(str(path))
                except UnicodeDecodeError:
                    continue
    return affected


def run_neighbor_stage(
    run_directory: Path,
    step3_directory: Path,
    step4_directory: Path,
    bootstrap_replicates: int = N_BOOTSTRAP,
    random_baseline_repeats: int = N_RANDOM_BASELINE_REPEATS,
    permutation_replicates: int = N_PERMUTATIONS,
    primary_k: int = PRIMARY_K_NEIGHBORS,
    sensitivity_k: tuple[int, int] = SENSITIVITY_K_NEIGHBORS,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    run_directory = run_directory.resolve()
    output_directory = run_directory / "neighbor_sharing"
    if not run_directory.exists():
        raise FileNotFoundError(f"Step 7 run does not exist: {run_directory}")
    gate_manifest_path = run_directory / "step7_manifest.json"
    gate_manifest = json.loads(gate_manifest_path.read_text())
    if gate_manifest.get("gate1_status") != "GO":
        raise RuntimeError("Gate 1 was not GO")
    if gate_manifest.get("neighbor_stage", {}).get("status") == "GATE2_COMPLETE":
        raise RuntimeError("Gate 2 is already complete for this run")
    if primary_k != PRIMARY_K_NEIGHBORS:
        raise RuntimeError("Primary k differs from the frozen value 10")
    if tuple(sensitivity_k) != SENSITIVITY_K_NEIGHBORS:
        raise RuntimeError("Sensitivity k values differ from 5 and 20")
    if random_baseline_repeats != N_RANDOM_BASELINE_REPEATS:
        raise RuntimeError("Random baseline repeats differ from 2,000")
    representation_path = (
        step4_directory / "test_participant_representations.parquet"
    )
    feature_path = step4_directory / "test_glycemic_nuisance_features.parquet"
    pipeline_root = step3_directory / "frozen_validation_pipeline"
    participant_ids, scores, projection_metadata = load_frozen_scores(
        representation_path, pipeline_root
    )
    clinical = load_clinical_frame(participant_ids, feature_path)
    sites = clinical["clinical_site"].astype(str).to_numpy()
    groups = clinical["study_group"].astype(str).to_numpy()
    k_values = (primary_k, *tuple(sensitivity_k))
    graphs, graph_edges = build_graphs(
        participant_ids, scores, clinical, k_values
    )
    graph_path = output_directory / "neighbor_graph_edges.parquet"
    atomic_parquet(graph_edges, graph_path)
    expected_graph_rows = len(CONDITIONS) * sum(k_values) * len(participant_ids)
    if len(graph_edges) != expected_graph_rows:
        raise RuntimeError("Neighbour graph size does not match frozen settings")

    participant_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    permutation_cache: dict[tuple[str, str], np.ndarray] = {}
    for condition in CONDITIONS:
        for k_neighbors in k_values:
            neighbors = graphs[(condition, k_neighbors)]
            for variable, variable_label, variable_type in VARIABLES:
                raw_values = clinical[variable].to_numpy()
                if variable_type == "continuous":
                    numeric_values = pd.to_numeric(
                        clinical[variable], errors="coerce"
                    ).to_numpy(float)
                    observed = np.isfinite(numeric_values)
                    standard_deviation = np.std(
                        numeric_values[observed], ddof=1
                    )
                    permutation_values = np.full(len(numeric_values), np.nan)
                    permutation_values[observed] = (
                        numeric_values[observed]
                        - np.mean(numeric_values[observed])
                    ) / standard_deviation
                else:
                    observed = clinical[variable].notna().to_numpy()
                    codes, _ = pd.factorize(clinical[variable], sort=True)
                    permutation_values = codes.astype(float)
                (
                    focal_rows,
                    unrestricted_weights,
                    site_weights,
                    insufficient_site,
                ) = participant_metrics_and_weights(
                    raw_values,
                    variable_type,
                    sites,
                    neighbors,
                    condition,
                    variable,
                    k_neighbors,
                    random_baseline_repeats,
                    seed,
                )
                focal_frame = pd.DataFrame(focal_rows)
                if focal_frame.empty:
                    raise RuntimeError(
                        f"No eligible focal rows for {condition} {variable}"
                    )
                focal_frame["participant_id"] = [
                    participant_ids[index]
                    for index in focal_frame["focal_index"]
                ]
                focal_frame["condition"] = condition
                focal_frame["k_neighbors"] = k_neighbors
                focal_frame["variable"] = variable
                focal_frame["clinical_site"] = [
                    sites[index] for index in focal_frame["focal_index"]
                ]
                focal_frame["study_group"] = [
                    groups[index] for index in focal_frame["focal_index"]
                ]
                participant_rows.extend(
                    focal_frame.drop(columns=["focal_index"]).to_dict("records")
                )
                observed_indices = np.flatnonzero(observed)
                global_key = (variable, "global")
                if global_key not in permutation_cache:
                    permutation_cache[global_key] = permuted_values(
                        permutation_values,
                        observed,
                        sites,
                        permutation_replicates,
                        stable_seed(
                            variable, "global_permutation", base_seed=seed
                        ),
                        within_site=False,
                    )
                site_key = (variable, "within_site")
                site_counts = pd.Series(sites[observed]).value_counts()
                within_site_available = bool((site_counts >= 2).all())
                if site_key not in permutation_cache:
                    permutation_cache[site_key] = permuted_values(
                        permutation_values,
                        observed,
                        sites,
                        permutation_replicates,
                        stable_seed(
                            variable,
                            "within_site_permutation",
                            base_seed=seed,
                        ),
                        within_site=within_site_available,
                    )
                result_rows.append(
                    summarize_result(
                        focal_frame,
                        variable,
                        variable_label,
                        variable_type,
                        condition,
                        k_neighbors,
                        "unrestricted_non_neighbours",
                        insufficient_site,
                        unrestricted_weights,
                        permutation_cache[global_key],
                        observed_indices,
                        bootstrap_replicates,
                        permutation_replicates,
                        seed,
                        "global_label_permutation",
                    )
                )
                result_rows.append(
                    summarize_result(
                        focal_frame,
                        variable,
                        variable_label,
                        variable_type,
                        condition,
                        k_neighbors,
                        "site_matched",
                        insufficient_site,
                        site_weights,
                        permutation_cache[site_key],
                        observed_indices,
                        bootstrap_replicates,
                        permutation_replicates,
                        seed,
                        "within_site_label_permutation"
                        if within_site_available
                        else "global_label_permutation_fallback",
                    )
                )
    participant_results = pd.DataFrame(participant_rows)
    required_participant_columns = [
        "participant_id",
        "condition",
        "k_neighbors",
        "variable",
        "neighbor_metric",
        "random_metric",
        "sharing_gain",
        "site_matched_random_metric",
        "site_matched_sharing_gain",
        "n_valid_neighbors",
        "n_valid_random_candidates",
        "clinical_site",
        "study_group",
    ]
    if not set(required_participant_columns).issubset(
        participant_results.columns
    ):
        raise RuntimeError("Participant output schema is incomplete")
    participant_path = (
        output_directory / "neighbor_sharing_by_participant.parquet"
    )
    atomic_parquet(participant_results, participant_path)
    results = add_fdr(pd.DataFrame(result_rows))
    result_path = output_directory / "neighbor_sharing_tier1_results.csv"
    results.to_csv(result_path, index=False)
    figure_path = (
        output_directory
        / "figure_neighbor_clinical_sharing_full_vs_neutral.png"
    )
    make_forest_figure(results, figure_path)
    report_path = output_directory / "neighbor_sharing_report.md"
    make_report(results, report_path)
    primary_neutral = results[
        (results["condition"] == "neutral_all")
        & (results["k_neighbors"] == primary_k)
        & (~results["site_matched"])
    ].copy()
    supported_primary = primary_neutral[
        (primary_neutral["bootstrap_ci_low"] > 0)
        & (primary_neutral["permutation_p"] < 0.05)
        & (primary_neutral["fdr_q"] < 0.05)
    ]["variable_label"].tolist()
    warnings: list[str] = []
    if results.loc[
        results["site_matched"], "n_insufficient_site_candidates"
    ].max() > 0:
        warnings.append(
            "Some focal participants had fewer than k eligible site-matched "
            "non-neighbours. All available candidates were used without "
            "replacement and the counts are reported."
        )
    blockers: list[str] = []
    output_hashes = {
        path.name: sha256_file(path)
        for path in (
            graph_path,
            participant_path,
            result_path,
            figure_path,
            report_path,
        )
    }
    gate_manifest["neighbor_stage"] = {
        "status": "GATE2_COMPLETE",
        "participant_count": len(participant_ids),
        "representation_spaces": projection_metadata,
        "k_values": list(k_values),
        "distance_metric": DISTANCE_METRIC,
        "directed_graph_rows": len(graph_edges),
        "random_baseline": {
            "repeats": random_baseline_repeats,
            "replacement": SAMPLE_WITH_REPLACEMENT,
            "types": [
                "unrestricted_non_neighbours",
                "site_matched",
            ],
        },
        "bootstrap": {
            "unit": "focal_participant",
            "replicates": bootstrap_replicates,
        },
        "permutation": {
            "replicates": permutation_replicates,
            "global_primary": True,
            "within_site_sensitivity": True,
        },
        "fdr_families": {
            "primary": "neutral_all, k=10, exactly nine Tier 1 variables",
            "secondary": "full_all, k=10, exactly nine Tier 1 variables",
        },
        "supported_primary_variables": supported_primary,
        "output_paths": {
            "tier1_results": str(result_path),
            "participant_results": str(participant_path),
            "graph_edges": str(graph_path),
            "forest_figure": str(figure_path),
            "report": str(report_path),
        },
        "output_hashes": output_hashes,
        "warnings": warnings,
        "blockers": blockers,
        "gate2_status": "GO" if not blockers else "NO-GO",
    }
    gate_manifest["latest_pointer_created"] = False
    write_json(gate_manifest_path, gate_manifest)
    step7_report = run_directory / "step7_report.md"
    step7_report.write_text(
        "# Step 7 closing pass\n\n"
        "Gate 1 schema verification and Gate 2 nearest-neighbour clinical "
        "sharing are complete. See schema_audit/schema_audit_report.md and "
        "neighbor_sharing/neighbor_sharing_report.md. Later stages have not "
        "run.\n"
    )
    with (run_directory / "step7_run.log").open("a") as handle:
        handle.write("STEP 7 neighbour stage completed\n")
        handle.write(f"Participant count: {len(participant_ids)}\n")
        handle.write(f"Directed graph rows: {len(graph_edges)}\n")
        handle.write(
            "Stopped before figures, HbA1c probes, and text as required.\n"
        )
    em_dash_files = scan_em_dash(
        [
            Path(__file__),
            output_directory,
            gate_manifest_path,
            step7_report,
            run_directory / "step7_run.log",
        ]
    )
    if em_dash_files:
        raise RuntimeError(
            "Generated text contains forbidden Unicode U+2014: "
            + ", ".join(em_dash_files)
        )
    return {
        "run_directory": str(run_directory),
        "neighbor_sharing_output_directory": str(output_directory),
        "eligible_participant_count": len(participant_ids),
        "directed_graph_rows": len(graph_edges),
        "primary_k": primary_k,
        "tier1_results_path": str(result_path),
        "participant_results_path": str(participant_path),
        "graph_path": str(graph_path),
        "forest_figure_path": str(figure_path),
        "report_path": str(report_path),
        "supported_primary_variables": supported_primary,
        "warnings": warnings,
        "blockers": blockers,
        "gate2_status": "GO" if not blockers else "NO-GO",
    }
