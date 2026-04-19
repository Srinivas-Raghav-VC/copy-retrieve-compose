from __future__ import annotations

from typing import Any, Dict, Iterable, List


def pair_matrix_mode(
    pairs: Iterable[str],
    locked_pairs: Iterable[str],
    substitution_plan: Iterable[dict] | None = None,
) -> str:
    provided = tuple(str(p).strip() for p in pairs)
    locked = tuple(str(p).strip() for p in locked_pairs)
    substitutions = list(substitution_plan or [])

    if provided == locked:
        return "locked"
    if len(provided) == len(locked) and substitutions:
        return "preapproved_substitution"
    if len(provided) < len(locked):
        return "subset_exploratory"
    if len(provided) > len(locked):
        return "expanded_exploratory"
    return "custom_matrix_exploratory"


def dataset_provenance_status(dataset_manifest: Dict[str, Any] | None) -> Dict[str, Any]:
    pair_manifests = {}
    if isinstance(dataset_manifest, dict):
        raw_pair_manifests = dataset_manifest.get("pair_manifests", {})
        if isinstance(raw_pair_manifests, dict):
            pair_manifests = raw_pair_manifests

    builtin_only_pairs: List[str] = []
    external_source_pairs: List[str] = []
    sources_by_pair: Dict[str, List[str]] = {}

    for pair_id, payload in pair_manifests.items():
        sources = payload.get("sources", []) if isinstance(payload, dict) else []
        source_names: List[str] = []
        has_external = False
        for source in sources:
            if not isinstance(source, dict):
                continue
            name = str(source.get("name", "")).strip()
            if not name:
                continue
            source_names.append(name)
            if name != "config_multiscript":
                has_external = True
        sources_by_pair[str(pair_id)] = source_names
        if has_external:
            external_source_pairs.append(str(pair_id))
        else:
            builtin_only_pairs.append(str(pair_id))

    return {
        "dataset_manifest_present": bool(pair_manifests),
        "all_pairs_have_external_sources": bool(pair_manifests) and not builtin_only_pairs,
        "builtin_only_pairs": sorted(builtin_only_pairs),
        "external_source_pairs": sorted(external_source_pairs),
        "sources_by_pair": sources_by_pair,
    }


def evaluate_protocol_compliance(
    *,
    pairs: Iterable[str],
    locked_pairs: Iterable[str],
    substitution_plan: Iterable[dict] | None,
    allow_underpowered_pairs: bool,
    enforce_pair_readiness: bool,
    substitution_audit: Dict[str, Any] | None,
    dataset_manifest: Dict[str, Any] | None,
) -> Dict[str, Any]:
    mode = pair_matrix_mode(
        pairs=pairs,
        locked_pairs=locked_pairs,
        substitution_plan=substitution_plan,
    )
    notes: List[str] = []
    confirmatory_protocol_passed = mode in {"locked", "preapproved_substitution"}

    if mode not in {"locked", "preapproved_substitution"}:
        notes.append(
            f"Pair matrix mode '{mode}' is exploratory and cannot support confirmatory claims."
        )
    if bool(allow_underpowered_pairs):
        confirmatory_protocol_passed = False
        notes.append("allow_underpowered_pairs=True is exploratory.")
    if not bool(enforce_pair_readiness):
        confirmatory_protocol_passed = False
        notes.append("Pair-readiness enforcement disabled.")

    substitution_policy_ok = True
    decisions = []
    if isinstance(substitution_audit, dict):
        raw_decisions = substitution_audit.get("decisions", [])
        if isinstance(raw_decisions, list):
            decisions = raw_decisions
    if mode == "preapproved_substitution":
        if not decisions:
            substitution_policy_ok = False
        else:
            substitution_policy_ok = all(
                bool(d.get("allowed_by_policy"))
                for d in decisions
                if isinstance(d, dict)
            )
        if not substitution_policy_ok:
            confirmatory_protocol_passed = False
            notes.append("Substitution policy audit did not approve requested substitution.")

    provenance = dataset_provenance_status(dataset_manifest)
    external_provenance_ok = bool(provenance["all_pairs_have_external_sources"])
    if not external_provenance_ok:
        confirmatory_protocol_passed = False
        missing_pairs = ", ".join(provenance["builtin_only_pairs"]) or "all pairs"
        notes.append(
            "External provenance requirement failed; builtin-only or missing-source pairs: "
            + missing_pairs
        )

    return {
        "pair_matrix_mode": mode,
        "confirmatory_protocol_passed": bool(confirmatory_protocol_passed),
        "substitution_policy_ok": bool(substitution_policy_ok),
        "external_provenance_ok": external_provenance_ok,
        "dataset_manifest_present": bool(provenance["dataset_manifest_present"]),
        "builtin_only_pairs": list(provenance["builtin_only_pairs"]),
        "external_source_pairs": list(provenance["external_source_pairs"]),
        "sources_by_pair": dict(provenance["sources_by_pair"]),
        "notes": notes,
    }
