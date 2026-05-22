#!/usr/bin/env python3
"""
07_llm_disagreement_study.py

LLM disagreement study vs deterministic Declare conformance.

Update (sampling policy)
------------------------
- Select a subset of CaseIDs equal to SAMPLE_FRACTION of the total cases.
- Keep the global CAP (MAX_TOTAL_PAIRS). If the sampled pool would exceed the CAP,
  the CAP limits the final number of (case, rule) pairs.

Sampling logic (per rule, balanced where possible)
--------------------------------------------------
- For each constraint/rule, sample up to K violations and K compliants
  BUT ONLY from the sampled CaseID subset.
- Final pairs are shuffled and truncated to MAX_TOTAL_PAIRS.

Outputs
-------
- llm_disagreement/llm_disagreement_results.csv
- llm_disagreement/llm_disagreement_summary.csv
- llm_disagreement/llm_disagreement_per_rule.csv
- llm_disagreement/llm_disagreement_mismatches.csv
- llm_disagreement/llm_disagreement_run_metadata.csv   (NEW)
- llm_disagreement/rules_used.txt                      (NEW)

Requirements
------------
pip install openai python-dotenv pandas
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# =============================
# CONFIGURATION
# =============================

CSV_SEP = ";"

# Inputs
CONFORMANCE_PATH = Path("conformance_results/conformance_results.csv")
GOLD_RULES_PATH = Path("event_rules/workflow_rules.csv")
LOG_PATH = Path("event_log/curia_log_en.csv")

# Impact ranking (to select top-N rules)
IMPACT_PATH = Path("conformance_impact/conformance_impact.csv")
N_RULES = 3  # top rules to evaluate (from IMPACT_PATH)

# Sampling knobs
SAMPLES_PER_RULE_PER_CLASS = 30  # K compliant + K violating per rule (if available)
MAX_PAIRS_PER_RULE = 2 * SAMPLES_PER_RULE_PER_CLASS  # hard limit per rule (safety)

# Case sampling: compare on a fraction of total cases
SAMPLE_FRACTION = 1 / 3  # 0.333...

# Global CAP: if exceeded, trim to this many (case, rule) pairs
MAX_TOTAL_PAIRS = 800

# Output
OUT_DIR = Path("llm_disagreement")
RESULTS_PATH = OUT_DIR / "llm_disagreement_results.csv"
SUMMARY_PATH = OUT_DIR / "llm_disagreement_summary.csv"
PER_RULE_PATH = OUT_DIR / "llm_disagreement_per_rule.csv"
MISMATCHES_PATH = OUT_DIR / "llm_disagreement_mismatches.csv"
RUN_META_PATH = OUT_DIR / "llm_disagreement_run_metadata.csv"  # NEW
RULES_USED_PATH = OUT_DIR / "rules_used.txt"  # NEW

# OpenAI
MODEL = "gpt-5.4"
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 700

# Study design
SEED = 7
MAX_EVENTS_IN_PROMPT = 45

# Declare4Py export convention
VIOLATION_VALUE = 1  # 1 = violation, 0 = compliant

# Output formatting
FLOAT_PRECISION = 3

# =============================
# HELPERS: loading + cleaning
# =============================

def _clean_constraint_name(name: str) -> str:
    """
    Normalize a Declare constraint header exported by Declare4Py.

    Args:
        name: Raw constraint column name.

    Returns:
        Cleaned constraint name without trailing placeholder noise.
    """
    # e.g., "Precedence[Opinion, Judgment] | | |" -> "Precedence[Opinion, Judgment]"
    return str(name).split("|", 1)[0].strip()


def round_float_columns(df: pd.DataFrame, precision: int) -> pd.DataFrame:
    """
    Round all float columns in a DataFrame to the specified precision.
    """
    df = df.copy()
    float_cols = df.select_dtypes(include=["float", "float32", "float64"]).columns
    df[float_cols] = df[float_cols].round(precision)
    return df


def load_conformance_matrix(path: Path) -> pd.DataFrame:
    """
    Load and normalize the deterministic conformance matrix.

    Args:
        path: Path to conformance CSV exported by the pipeline.

    Returns:
        DataFrame indexed by CaseID with integer 0/1 values per constraint.
    """
    df = pd.read_csv(path, sep=CSV_SEP)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={c: _clean_constraint_name(c) for c in df.columns})

    # Require CaseID now that 04_ exports it
    if "CaseID" in df.columns:
        df["CaseID"] = df["CaseID"].astype(str).str.strip()
        df = df.set_index("CaseID")
    else:
        # fallback (should not happen now): create synthetic ids
        df.index = [f"case_{i:06d}" for i in range(1, len(df) + 1)]

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=1, how="all").fillna(0).astype(int)

    df.index = df.index.astype(str).str.strip()
    print(f"Loaded conformance matrix with {len(df)} cases and {df.shape[1]} rules.")
    return df


def load_gold_rules(path: Path) -> pd.DataFrame:
    """
    Load GOLD workflow rules metadata from CSV.

    Args:
        path: Path to GOLD rules CSV file.

    Returns:
        DataFrame with cleaned column names.
    """
    df = pd.read_csv(path, sep=CSV_SEP)
    df.columns = df.columns.str.strip()
    return df


def load_log(path: Path) -> pd.DataFrame:
    """
    Load and validate the event log used to build case traces.

    Args:
        path: Path to event-log CSV.

    Returns:
        Sorted DataFrame with normalized `CaseID`, `Activity`, and parsed
        `Timestamp` columns.

    Raises:
        ValueError: If required columns are missing or timestamps are invalid.
    """
    df = pd.read_csv(path, sep=CSV_SEP)
    df.columns = df.columns.str.strip()

    required = {"CaseID", "Activity", "Timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"LOG CSV missing required columns: {sorted(missing)}")

    df["CaseID"] = df["CaseID"].astype(str).str.strip()
    df["Activity"] = df["Activity"].astype(str).str.strip()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    if df["Timestamp"].isna().any():
        bad = df[df["Timestamp"].isna()][["CaseID", "Activity", "Timestamp"]].head(10)
        raise ValueError("Some Timestamp values could not be parsed.\n" + bad.to_string(index=False))

    df = df.sort_values(["CaseID", "Timestamp", "Activity"], kind="mergesort").reset_index(drop=True)
    return df


def select_top_rules_from_impact(
    impact_path: Path,
    *,
    n_rules: int,
    csv_sep: str,
    available_rules: List[str],
) -> List[str]:
    """
    Read conformance_impact.csv and return top-N rules (constraint names) to evaluate.

    Args:
        impact_path: Path to impact-ranking CSV.
        n_rules: Number of top rules to select.
        csv_sep: CSV separator used in the file.
        available_rules: Constraint names available in current conformance data.

    Returns:
        List of selected constraint names.

    Raises:
        ValueError: If expected columns are missing or selection is empty after
            filtering on available rules.
    """
    impact_df = pd.read_csv(impact_path, sep=csv_sep)
    impact_df.columns = impact_df.columns.str.strip()

    if "rule" not in impact_df.columns:
        raise ValueError("conformance_impact.csv must contain a 'rule' column with constraint names.")

    # Prefer sorting by impact_delta if present
    if "impact_delta" in impact_df.columns:
        impact_df["impact_delta"] = pd.to_numeric(impact_df["impact_delta"], errors="coerce")
        impact_df = impact_df.sort_values("impact_delta", ascending=False)

    top = impact_df["rule"].astype(str).str.strip().head(n_rules).tolist()

    # Keep only rules that exist in current conformance matrix
    top = [r for r in top if r in available_rules]

    if not top:
        raise ValueError(
            "Top rules list is empty after filtering. "
            "Check that conformance_impact.csv 'rule' values match df_conf columns."
        )

    return top


# =============================
# TRACE formatting
# =============================

def build_traces(df_log: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Build per-case trace tables from the normalized event log.

    Args:
        df_log: Event log DataFrame with `CaseID`, `Timestamp`, and `Activity`.

    Returns:
        Dictionary mapping CaseID to ordered trace DataFrame.
    """
    traces: Dict[str, pd.DataFrame] = {}
    for case_id, g in df_log.groupby("CaseID", sort=True):
        traces[case_id] = g[["Timestamp", "Activity"]].sort_values(["Timestamp", "Activity"], kind="mergesort")
    return traces


def trace_to_text(trace_df: pd.DataFrame, max_events: int) -> str:
    """
    Convert a case trace DataFrame into prompt-friendly plain text.

    Long traces are truncated by keeping head and tail events with an explicit
    truncation marker.

    Args:
        trace_df: Single-case trace DataFrame.
        max_events: Maximum number of events to render.

    Returns:
        Multi-line text representation of the trace.
    """
    if trace_df is None or trace_df.empty:
        return "(empty trace)"

    td = trace_df.copy()

    if len(td) > max_events:
        head_n = max_events // 2
        tail_n = max_events - head_n
        head = td.head(head_n)
        tail = td.tail(tail_n)

        lines = [f"{r.Timestamp.isoformat(sep=' ')} | {r.Activity}" for r in head.itertuples(index=False)]
        lines.append("... (trace truncated) ...")
        lines.extend([f"{r.Timestamp.isoformat(sep=' ')} | {r.Activity}" for r in tail.itertuples(index=False)])
        return "\n".join(lines)

    return "\n".join([f"{r.Timestamp.isoformat(sep=' ')} | {r.Activity}" for r in td.itertuples(index=False)])


# =============================
# GOLD matching (best-effort)
# =============================

_CONSTRAINT_RE = re.compile(r"^(?P<tpl>[A-Za-z_]+)\[(?P<a>.*?),(?P<b>.*?)\]\s*$")


def parse_constraint(constraint_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse a binary Declare constraint string.

    Args:
        constraint_name: Constraint text like `Response[A, B]`.

    Returns:
        Tuple `(template, A, B)` or `(None, None, None)` if parsing fails.
    """
    m = _CONSTRAINT_RE.match(constraint_name.strip())
    if not m:
        return None, None, None
    return m.group("tpl").strip(), m.group("a").strip(), m.group("b").strip()


def lookup_gold_for_constraint(constraint_name: str, gold_df: pd.DataFrame) -> Dict[str, str]:
    """
    Retrieve GOLD metadata for a constraint by matching trigger/related concepts.

    Args:
        constraint_name: Declare constraint name.
        gold_df: GOLD rules DataFrame.

    Returns:
        Metadata dictionary with GOLD fields; values are empty when no match is
        available or required columns are missing.
    """
    tpl, a, b = parse_constraint(constraint_name)
    if tpl is None:
        return {
            "gold_rule_id": "",
            "gold_rule_name": "",
            "gold_nl_rule": "",
            "gold_excerpt_reference": "",
            "gold_assumptions": "",
        }

    required = {
        "rule_id",
        "name",
        "trigger_concept",
        "related_concept",
        "natural_language_rule",
        "excerpt_reference",
        "assumptions",
    }
    if not required.issubset(set(gold_df.columns)):
        return {
            "gold_rule_id": "",
            "gold_rule_name": "",
            "gold_nl_rule": "",
            "gold_excerpt_reference": "",
            "gold_assumptions": "",
        }

    g = gold_df.copy()
    g["trigger_concept"] = g["trigger_concept"].astype(str).str.strip()
    g["related_concept"] = g["related_concept"].astype(str).str.strip()

    match = g[(g["trigger_concept"] == a) & (g["related_concept"] == b)]
    if match.empty:
        return {
            "gold_rule_id": "",
            "gold_rule_name": "",
            "gold_nl_rule": "",
            "gold_excerpt_reference": "",
            "gold_assumptions": "",
        }

    r = match.iloc[0]
    return {
        "gold_rule_id": str(r["rule_id"]),
        "gold_rule_name": str(r["name"]),
        "gold_nl_rule": str(r["natural_language_rule"]),
        "gold_excerpt_reference": str(r["excerpt_reference"]),
        "gold_assumptions": str(r["assumptions"]) if pd.notna(r["assumptions"]) else "",
    }


# =============================
# Sampling
# =============================

def sample_case_ids(
    all_case_ids: List[str],
    *,
    fraction: float,
    seed: int,
) -> Set[str]:
    """
    Sample a subset of CaseIDs equal to 'fraction' of the total (at least 1).
    """
    rnd = random.Random(seed)
    n_total = len(all_case_ids)
    n_sample = int(n_total * fraction)
    n_sample = max(1, n_sample)
    n_sample = min(n_total, n_sample)
    return set(rnd.sample(all_case_ids, k=n_sample))


def sample_pairs_balanced_per_rule(
    df_conf: pd.DataFrame,
    eligible_case_ids: Set[str],
    *,
    samples_per_rule_per_class: int,
    seed: int,
    max_total_pairs: int,
    max_pairs_per_rule: int,
) -> List[Tuple[str, str, int]]:
    """
    Balanced per rule: K violations + K compliants where possible, but only from eligible_case_ids.
    Returns list of (case_id, constraint, det_label) where det_label is 0/1.
    """
    rnd = random.Random(seed)
    pairs: List[Tuple[str, str, int]] = []

    eligible_index = df_conf.index.intersection(list(eligible_case_ids))

    for constraint in df_conf.columns:
        col = df_conf.loc[eligible_index, constraint]

        viol_cases = col[col == VIOLATION_VALUE].index.tolist()
        ok_cases = col[col != VIOLATION_VALUE].index.tolist()

        if not viol_cases or not ok_cases:
            print(f"[{constraint}] skipped (viol={len(viol_cases)}, ok={len(ok_cases)})")
            continue

        k_v = min(samples_per_rule_per_class, len(viol_cases))
        k_o = min(samples_per_rule_per_class, len(ok_cases))

        # enforce a hard per-rule limit (safety)
        per_rule_cap_each = max_pairs_per_rule // 2
        k_v = min(k_v, per_rule_cap_each)
        k_o = min(k_o, per_rule_cap_each)

        print(
            f"[{constraint}] available: viol={len(viol_cases)}, ok={len(ok_cases)} | "
            f"sampled: viol={k_v}, ok={k_o}"
        )

        sampled_v = rnd.sample(viol_cases, k=k_v)
        sampled_o = rnd.sample(ok_cases, k=k_o)

        pairs.extend([(c, constraint, 1) for c in sampled_v])
        pairs.extend([(c, constraint, 0) for c in sampled_o])

        if len(pairs) >= max_total_pairs:
            break

    rnd.shuffle(pairs)
    return pairs[:max_total_pairs]


# =============================
# LLM prompt + call
# =============================

def build_llm_prompt(
    case_id: str,
    constraint: str,
    gold_meta: Dict[str, str],
    trace_text: str,
) -> str:
    """
    Build the LLM prompt for one (case, constraint) evaluation.

    Args:
        case_id: Case identifier.
        constraint: Declare-style constraint name.
        gold_meta: GOLD metadata used to contextualize the rule.
        trace_text: Plain-text representation of the case trace.

    Returns:
        Prompt string for the Responses API call.
    """
    return f"""
You are a legal process compliance analyst.

TASK
Given:
1) a workflow compliance rule (natural language + Declare-style constraint name),
2) a case trace (timestamp | activity),
decide whether the case is COMPLIANT with the rule.

IMPORTANT
- You must base your decision ONLY on the provided trace and the rule text below.
- If the trace does not contain enough information to decide, answer "uncertain" and explain why.
- Return ONLY valid JSON (no prose).

OUTPUT JSON KEYS
- case_id: string
- constraint: string
- gold_rule_id: string (may be empty)
- llm_label: one of ["compliant","violation","uncertain"]
- confidence: number between 0 and 1
- rationale: string (2-4 sentences, plain English)
- evidence_lines: array of up to 3 strings copied from the trace that support your judgement

RULE
- Declare constraint name: {constraint}
- GOLD rule id: {gold_meta.get("gold_rule_id","")}
- GOLD rule name: {gold_meta.get("gold_rule_name","")}
- GOLD natural-language rule: {gold_meta.get("gold_nl_rule","")}
- Excerpt reference: {gold_meta.get("gold_excerpt_reference","")}
- Assumptions: {gold_meta.get("gold_assumptions","")}

CASE TRACE
{trace_text}
""".strip()


def _safe_usage_total_tokens(resp: Any) -> int:
    """
    Try to extract total tokens from an OpenAI Responses API response.
    Returns 0 if not available.
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0

    # SDKs may expose usage as object or dict-like
    if isinstance(usage, dict):
        return int(usage.get("total_tokens", 0) or 0)

    return int(getattr(usage, "total_tokens", 0) or 0)


def call_llm_json(client: OpenAI, prompt: str) -> Tuple[Dict[str, Any], int]:
    """
    Call the model and parse JSON output defensively.

    Args:
        client: OpenAI client instance.
        prompt: Prompt text sent to the model.

    Returns:
        Tuple `(parsed_json, tokens_used_for_call)`.
        On invalid JSON, a fallback payload with `llm_label='uncertain'` is
        returned.
    """
    resp = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    tokens_used = _safe_usage_total_tokens(resp)

    text = resp.output_text.strip()
    try:
        return json.loads(text), tokens_used
    except json.JSONDecodeError:
        return (
            {
                "case_id": None,
                "constraint": None,
                "gold_rule_id": "",
                "llm_label": "uncertain",
                "confidence": 0.0,
                "rationale": f"Invalid JSON from model. Raw: {text[:300]}",
                "evidence_lines": [],
                "_raw_model_output": text,
            },
            tokens_used,
        )


def llm_label_to_binary(llm_label: str) -> Optional[int]:
    """
    Map textual LLM labels to deterministic binary convention.

    Args:
        llm_label: One of `violation`, `compliant`, `uncertain` (case-insensitive).

    Returns:
        `1` for violation, `0` for compliant, `None` for uncertain/unknown.
    """
    llm_label = (llm_label or "").strip().lower()
    if llm_label == "violation":
        return 1
    if llm_label == "compliant":
        return 0
    return None  # uncertain


# =============================
# Metrics
# =============================

def compute_summary(
    df_res: pd.DataFrame,
    *,
    n_cases_total: int,
    n_cases_sampled: int,
    sample_fraction: float,
    top_k_rules_used: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute global and per-rule disagreement summary tables.

    Args:
        df_res: Row-level comparison results between deterministic and LLM labels.
        n_cases_total: Number of cases in the full conformance matrix.
        n_cases_sampled: Number of sampled eligible cases.
        sample_fraction: Sampling fraction used for eligible cases.
        top_k_rules_used: Number of top impact rules evaluated.

    Returns:
        Tuple `(summary_df, per_rule_df)` with aggregated metrics.
    """
    df = df_res.copy()
    df["llm_bin"] = df["llm_label"].apply(llm_label_to_binary)
    df["is_uncertain"] = df["llm_bin"].isna()

    strict = df[~df["is_uncertain"]].copy()
    strict["agree"] = (strict["llm_bin"] == strict["det_label"]).astype(int)

    tp = int(((strict["det_label"] == 1) & (strict["llm_bin"] == 1)).sum())
    tn = int(((strict["det_label"] == 0) & (strict["llm_bin"] == 0)).sum())
    fp = int(((strict["det_label"] == 0) & (strict["llm_bin"] == 1)).sum())
    fn = int(((strict["det_label"] == 1) & (strict["llm_bin"] == 0)).sum())

    summary = {
        "n_cases_total": int(n_cases_total),
        "n_cases_sampled": int(n_cases_sampled),
        "sample_fraction": float(sample_fraction),
        "top_k_rules_used": int(top_k_rules_used),
        "n_cases_in_results": int(df["case_id"].nunique()) if len(df) else 0,
        "n_total": int(len(df)),
        "n_strict": int(len(strict)),
        "uncertain_rate": float(df["is_uncertain"].mean()) if len(df) else 0.0,
        "agreement_rate_strict": float(strict["agree"].mean()) if len(strict) else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

    per_rule = (
        strict.groupby("constraint")
        .agg(
            n=("case_id", "count"),
            agreement=("agree", "mean"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
        .sort_values(["agreement", "n"], ascending=[True, False])
    )

    return pd.DataFrame([summary]), per_rule


# =============================
# MAIN
# =============================

def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO format with second precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    """
    Run the full LLM disagreement study pipeline.

    Steps:
    1. Load conformance, impact ranking, GOLD rules, and event log.
    2. Select top impact rules and sampled case IDs.
    3. Build balanced (case, rule) evaluation pairs.
    4. Query the LLM and collect row-level comparisons.
    5. Compute summaries and save all output artifacts.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Run metadata start ----
    start_ts = _utc_now_iso()
    start_dt = datetime.now(timezone.utc)

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found. Put it in your .env file.")
    client = OpenAI(api_key=api_key)

    print("> Loading inputs")
    df_conf = load_conformance_matrix(CONFORMANCE_PATH)

    # Restrict evaluation to top-N rules from impact ranking
    top_rules = select_top_rules_from_impact(
        IMPACT_PATH,
        n_rules=N_RULES,
        csv_sep=CSV_SEP,
        available_rules=df_conf.columns.tolist(),
    )
    df_conf = df_conf[top_rules]
    print(f"Using N_RULES={N_RULES} rules from conformance impact ranking: {len(top_rules)}")
    for r in top_rules:
        print(f"  - {r}")


    gold_df = load_gold_rules(GOLD_RULES_PATH)

    # Save rules used to a single file (NEW)
    # Save rules used to a single file (formal + GOLD natural language)
    rules_joined = " | ".join(top_rules)

    with open(RULES_USED_PATH, "w", encoding="utf-8") as f:
        f.write("Rules used (one per line)\n")
        f.write("\n".join(top_rules))
        f.write("\n\nRules used (joined)\n")
        f.write(rules_joined)
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("Rules used (formal + GOLD)\n\n")

        for i, constraint in enumerate(top_rules, start=1):
            gold_meta = lookup_gold_for_constraint(constraint, gold_df)
            f.write(f"RULE {i}\n")
            f.write(f"Constraint: {constraint}\n")
            f.write(f"GOLD_ID: {gold_meta.get('gold_rule_id','')}\n")
            f.write(f"GOLD_Name: {gold_meta.get('gold_rule_name','')}\n")
            f.write(f"GOLD_NL: {gold_meta.get('gold_nl_rule','')}\n")
            f.write(f"Excerpt: {gold_meta.get('gold_excerpt_reference','')}\n")
            f.write(f"Assumptions: {gold_meta.get('gold_assumptions','')}\n")
            f.write("\n" + "-" * 60 + "\n\n")

    df_log = load_log(LOG_PATH)
    traces = build_traces(df_log)

    # 1) Sample CaseIDs = 1/3 of total
    all_case_ids = df_conf.index.tolist()
    eligible_case_ids = sample_case_ids(all_case_ids, fraction=SAMPLE_FRACTION, seed=SEED)
    print(f"Eligible CaseIDs sampled: {len(eligible_case_ids)} / {len(all_case_ids)} (fraction={SAMPLE_FRACTION:.3f})")

    # 2) Sample pairs (balanced per rule, only among eligible CaseIDs), and capped
    pairs = sample_pairs_balanced_per_rule(
        df_conf,
        eligible_case_ids,
        samples_per_rule_per_class=SAMPLES_PER_RULE_PER_CLASS,
        seed=SEED,
        max_total_pairs=MAX_TOTAL_PAIRS,
        max_pairs_per_rule=MAX_PAIRS_PER_RULE,
    )
    print(f"Sampled (case, rule) pairs: {len(pairs)} (CAP={MAX_TOTAL_PAIRS})")

    # ---- LLM accounting ----
    llm_queries_executed = 0
    total_tokens_used = 0

    rows: List[Dict[str, Any]] = []
    for case_id, constraint, det_label in pairs:
        trace_df = traces.get(case_id)

        if trace_df is None:
            rows.append({
                "case_id": case_id,
                "constraint": constraint,
                "det_label": det_label,
                "llm_label": "uncertain",
                "confidence": 0.0,
                "rationale": "CaseID not found in log trace.",
                "evidence_lines": "[]",
                "gold_rule_id": "",
                "gold_rule_name": "",
                "gold_excerpt_reference": "",
                "gold_assumptions": "",
                "gold_nl_rule": "",
            })
            continue

        trace_text = trace_to_text(trace_df, max_events=MAX_EVENTS_IN_PROMPT)
        gold_meta = lookup_gold_for_constraint(constraint, gold_df)

        prompt = build_llm_prompt(case_id, constraint, gold_meta, trace_text)

        out, tokens_used = call_llm_json(client, prompt)
        llm_queries_executed += 1
        total_tokens_used += int(tokens_used)

        llm_label = str(out.get("llm_label", "uncertain")).strip().lower()
        conf = float(out.get("confidence", 0.0)) if out.get("confidence") is not None else 0.0
        rationale = str(out.get("rationale", "")).strip()

        ev = out.get("evidence_lines", [])
        ev_str = json.dumps(ev if isinstance(ev, list) else [], ensure_ascii=False)

        rows.append({
            "case_id": case_id,
            "constraint": constraint,
            "det_label": det_label,  # 1=violation, 0=compliant
            "llm_label": llm_label,
            "confidence": conf,
            "rationale": rationale,
            "evidence_lines": ev_str,
            "gold_rule_id": gold_meta.get("gold_rule_id", ""),
            "gold_rule_name": gold_meta.get("gold_rule_name", ""),
            "gold_excerpt_reference": gold_meta.get("gold_excerpt_reference", ""),
            "gold_assumptions": gold_meta.get("gold_assumptions", ""),
            "gold_nl_rule": gold_meta.get("gold_nl_rule", ""),
        })

    df_res = pd.DataFrame(rows)

    summary_df, per_rule_df = compute_summary(
        df_res,
        n_cases_total=len(all_case_ids),
        n_cases_sampled=len(eligible_case_ids),
        sample_fraction=SAMPLE_FRACTION,
        top_k_rules_used=len(top_rules),
    )

    df_res["llm_bin"] = df_res["llm_label"].apply(llm_label_to_binary)
    df_res["is_uncertain"] = df_res["llm_bin"].isna()
    df_res["agree_strict"] = ((df_res["llm_bin"] == df_res["det_label"]) & (~df_res["is_uncertain"])).astype(int)

    mismatches = df_res[(~df_res["is_uncertain"]) & (df_res["agree_strict"] == 0)].copy()

    # Round float columns
    df_res = round_float_columns(df_res, FLOAT_PRECISION)
    summary_df = round_float_columns(summary_df, FLOAT_PRECISION)
    per_rule_df = round_float_columns(per_rule_df, FLOAT_PRECISION)
    mismatches = round_float_columns(mismatches, FLOAT_PRECISION)

    df_res.to_csv(RESULTS_PATH, sep=CSV_SEP, index=False)
    summary_df.to_csv(SUMMARY_PATH, sep=CSV_SEP, index=False)
    per_rule_df.to_csv(PER_RULE_PATH, sep=CSV_SEP, index=False)
    mismatches.to_csv(MISMATCHES_PATH, sep=CSV_SEP, index=False)

    # ---- Run metadata end (NEW) ----
    end_ts = _utc_now_iso()
    end_dt = datetime.now(timezone.utc)
    delta_minutes = (end_dt - start_dt).total_seconds() / 60.0

    run_meta_df = pd.DataFrame([{
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "delta_minutes": round(delta_minutes, FLOAT_PRECISION),
        "llm_queries_executed": int(llm_queries_executed),
        "total_tokens_used": int(total_tokens_used),
    }])
    run_meta_df.to_csv(RUN_META_PATH, sep=CSV_SEP, index=False)

    print(f"Saved results: {RESULTS_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")
    print(f"Saved per-rule: {PER_RULE_PATH}")
    print(f"Saved mismatches: {MISMATCHES_PATH}")
    print(f"Saved run metadata: {RUN_META_PATH}")
    print(f"Saved rules used: {RULES_USED_PATH}")

    print("\n=== QUICK SUMMARY ===")
    print(summary_df.to_string(index=False))
    print("\n=== RUN METADATA ===")
    print(run_meta_df.to_string(index=False))


if __name__ == "__main__":
    main()