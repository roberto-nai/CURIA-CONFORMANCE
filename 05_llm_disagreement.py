#!/usr/bin/env python3
"""
05_llm_disagreement_study.py

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
pip install openai anthropic python-dotenv pandas
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import random
import re
import time
from functools import lru_cache
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
GOLD_RULES_PATH = Path("curia_rules/workflow_rules.csv") # event_rules
LOG_PATH = Path("event_log/curia_log_en.csv")
LLM_CONFIG_PATH = Path("llm_config.json")
LLM_PROMPTS_PATH = Path("llm_prompts.json")

# Impact ranking (to select top-N rules)
IMPACT_PATH = Path("conformance_impact/conformance_impact.csv")
N_RULES = 5  # top rules to evaluate (from IMPACT_PATH)

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


def csv_path_for_model(path: Path, model: str) -> Path:
    """Add a filesystem-safe model suffix before a CSV file extension."""
    safe_model = re.sub(r"[^A-Za-z0-9_-]+", "_", model).strip("_")
    if not safe_model:
        raise ValueError(f"Model name {model!r} cannot be converted to a safe filename.")
    return path.with_name(f"{path.stem}_{safe_model}{path.suffix}")


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


def load_llm_config(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load and minimally validate the model-keyed LLM configuration."""
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict) or not config:
        raise ValueError(f"{path} must contain a non-empty JSON object.")

    required = {
        "provider", "temperature", "top_p", "max_output_tokens", "seed",
        "response_format", "tools",
    }
    for model, settings in config.items():
        if not isinstance(settings, dict):
            raise ValueError(f"Configuration for model {model!r} must be an object.")
        missing = required - settings.keys()
        if missing:
            raise ValueError(f"Model {model!r} is missing settings: {sorted(missing)}")

        provider = str(settings["provider"]).strip().lower()
        seed = settings["seed"]
        if provider in {"openai", "anthropic"} and seed is not None:
            raise ValueError(
                f"Model {model!r} uses provider {settings['provider']!r}, "
                "which does not support seed in the API used by this script; "
                "set seed to null."
            )
        if provider == "ollama" and (
            isinstance(seed, bool) or not isinstance(seed, int)
        ):
            raise ValueError(
                f"Ollama model {model!r} requires seed to be an integer."
            )
    return config


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
            print(
                f"[{constraint}] skipped "
                f"(violations={len(viol_cases)}, compliant={len(ok_cases)})"
            )
            continue

        k_v = min(samples_per_rule_per_class, len(viol_cases))
        k_o = min(samples_per_rule_per_class, len(ok_cases))

        # enforce a hard per-rule limit (safety)
        per_rule_cap_each = max_pairs_per_rule // 2
        k_v = min(k_v, per_rule_cap_each)
        k_o = min(k_o, per_rule_cap_each)

        print(
            f"[{constraint}] available: violations={len(viol_cases)}, "
            f"compliant={len(ok_cases)} | sampled: violations={k_v}, compliant={k_o}"
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

@lru_cache(maxsize=None)
def load_llm_prompt(path: Path, prompt_key: str) -> str:
    """Load a prompt template identified by its key from a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        prompts = json.load(f)

    if not isinstance(prompts, dict):
        raise ValueError(f"{path} must contain a JSON object.")

    prompt = prompts.get(prompt_key)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Prompt key {prompt_key!r} is missing or empty in {path}.")
    return prompt

def build_llm_prompt(
    case_id: str,
    constraint: str,
    gold_meta: Dict[str, str],
    trace_text: str,
    prompt_key: str = "prompt_1",
) -> str:
    """
    Build the LLM prompt for one (case, constraint) evaluation.

    Args:
        case_id: Case identifier.
        constraint: Declare-style constraint name.
        gold_meta: GOLD metadata used to contextualize the rule.
        trace_text: Plain-text representation of the case trace.
        prompt_key: Key of the prompt to load from `llm_prompts.json`.

    Returns:
        Prompt string for the Responses API call.
    """
    prompt_template = load_llm_prompt(LLM_PROMPTS_PATH, prompt_key)
    return prompt_template.format(
        case_id=case_id,
        constraint=constraint,
        gold_rule_id=gold_meta.get("gold_rule_id", ""),
        gold_rule_name=gold_meta.get("gold_rule_name", ""),
        gold_nl_rule=gold_meta.get("gold_nl_rule", ""),
        gold_excerpt_reference=gold_meta.get("gold_excerpt_reference", ""),
        gold_assumptions=gold_meta.get("gold_assumptions", ""),
        trace_text=trace_text,
    ).strip()


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


def create_llm_client(provider: str) -> Any:
    """Create the native/provider-compatible client using environment variables."""
    provider_lower = provider.strip().lower()
    if provider_lower == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not found in the environment or .env file.")
        return OpenAI(api_key=api_key)

    if provider_lower == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not found in the environment or .env file.")
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError("Install the Anthropic SDK with: pip install anthropic") from exc
        return Anthropic(api_key=api_key)

    if provider_lower == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return OpenAI(api_key="ollama", base_url=base_url)

    if provider_lower == "meta":
        api_key = os.getenv("META_API_KEY")
        base_url = os.getenv("META_BASE_URL")
        if not api_key or not base_url:
            raise EnvironmentError(
                "META_API_KEY and META_BASE_URL are required for the Meta model. "
                "META_BASE_URL must point to an OpenAI-compatible inference endpoint."
            )
        return OpenAI(api_key=api_key, base_url=base_url)

    raise ValueError(f"Unsupported LLM provider: {provider!r}")


def parse_model_json(text: str) -> Dict[str, Any]:
    """Parse a JSON object, tolerating surrounding prose or Markdown fences."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as direct_error:
        object_start = text.find("{")
        if object_start < 0:
            raise direct_error
        try:
            payload, _ = json.JSONDecoder().raw_decode(text, object_start)
        except json.JSONDecodeError:
            raise direct_error

    if not isinstance(payload, dict):
        raise ValueError("Model output JSON must be an object.")
    return payload


def call_llm_json(
    client: Any,
    prompt: str,
    *,
    model: str,
    settings: Dict[str, Any],
) -> Tuple[Dict[str, Any], int]:
    """
    Call the model and parse JSON output defensively.

    Args:
        client: Provider-specific client instance.
        prompt: Prompt text sent to the model.
        model: Model identifier, taken from the JSON object key.
        settings: Provider and generation parameters loaded from JSON.

    Returns:
        Tuple `(parsed_json, tokens_used_for_call)`.
        On invalid JSON, a fallback payload with `llm_label='uncertain'` is
        returned.
    """
    provider = str(settings["provider"]).strip().lower()
    temperature = float(settings["temperature"])
    top_p = float(settings["top_p"])
    max_tokens = int(settings["max_output_tokens"])
    json_response = str(settings["response_format"]).strip().upper() == "JSON"

    if provider == "openai":
        kwargs: Dict[str, Any] = {
            "model": model,
            "input": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
        }
        if json_response:
            kwargs["text"] = {"format": {"type": "json_object"}}
        resp = client.responses.create(**kwargs)
        tokens_used = _safe_usage_total_tokens(resp)
        text = resp.output_text.strip()
    elif provider == "anthropic":
        # Anthropic models may reject requests containing both sampling
        # parameters. Use temperature and leave top_p at the model default.
        resp = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = getattr(resp, "usage", None)
        tokens_used = int(getattr(usage, "input_tokens", 0) or 0) + int(
            getattr(usage, "output_tokens", 0) or 0
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ).strip()
    elif provider in {"meta", "ollama"}:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if json_response:
            kwargs["response_format"] = {"type": "json_object"}
        if provider == "ollama":
            kwargs["seed"] = int(settings["seed"])
        resp = client.chat.completions.create(**kwargs)
        tokens_used = _safe_usage_total_tokens(resp)
        text = (resp.choices[0].message.content or "").strip()
    else:
        raise ValueError(f"Unsupported LLM provider: {settings['provider']!r}")

    try:
        return parse_model_json(text), tokens_used
    except (json.JSONDecodeError, ValueError):
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
    if "model" in df_res.columns:
        summary_parts: List[pd.DataFrame] = []
        per_rule_parts: List[pd.DataFrame] = []
        for model, model_df in df_res.groupby("model", sort=False):
            model_df = model_df.drop(columns=["model"])
            summary_part, per_rule_part = compute_summary(
                model_df,
                n_cases_total=n_cases_total,
                n_cases_sampled=n_cases_sampled,
                sample_fraction=sample_fraction,
                top_k_rules_used=top_k_rules_used,
            )
            summary_part.insert(0, "model", model)
            per_rule_part.insert(0, "model", model)
            summary_parts.append(summary_part)
            per_rule_parts.append(per_rule_part)
        return pd.concat(summary_parts, ignore_index=True), pd.concat(per_rule_parts, ignore_index=True)

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


def parse_cli_args() -> argparse.Namespace:
    """Parse and return the model and prompt selected from the command line."""
    parser = argparse.ArgumentParser(
        description="Compare one configured LLM against deterministic Declare conformance."
    )
    parser.add_argument(
        "model",
        help="Model key defined in llm_config.json.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="prompt_1",
        help="Prompt key defined in llm_prompts.json (default: prompt_1).",
    )
    return parser.parse_args()


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
    args = parse_cli_args()
    full_llm_config = load_llm_config(LLM_CONFIG_PATH)
    if args.model not in full_llm_config:
        available_models = ", ".join(full_llm_config.keys())
        raise SystemExit(
            f"Unknown model {args.model!r}. Available models: {available_models}"
        )

    try:
        load_llm_prompt(LLM_PROMPTS_PATH, args.prompt)
    except ValueError as exc:
        with LLM_PROMPTS_PATH.open("r", encoding="utf-8") as f:
            available_prompts = ", ".join(json.load(f).keys())
        raise SystemExit(
            f"Unknown prompt {args.prompt!r}. Available prompts: {available_prompts}"
        ) from exc

    llm_config = {args.model: full_llm_config[args.model]}
    results_path = csv_path_for_model(RESULTS_PATH, args.model)
    summary_path = csv_path_for_model(SUMMARY_PATH, args.model)
    per_rule_path = csv_path_for_model(PER_RULE_PATH, args.model)
    mismatches_path = csv_path_for_model(MISMATCHES_PATH, args.model)
    run_meta_path = csv_path_for_model(RUN_META_PATH, args.model)

    program_start_dt = datetime.now().astimezone()
    program_start_timer = time.perf_counter()
    print(f"Programme started: {program_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Run metadata start ----
    start_ts = _utc_now_iso()
    start_dt = datetime.now(timezone.utc)

    load_dotenv()
    clients = {
        model: create_llm_client(str(settings["provider"]))
        for model, settings in llm_config.items()
    }
    print(f"Selected model: {args.model}")
    print(f"Selected prompt: {args.prompt}")
    print()

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
    print()

    # 2) Sample pairs (balanced per rule, only among eligible CaseIDs), and capped
    print("> Sampling (case, rule) pairs for evaluation")
    pairs = sample_pairs_balanced_per_rule(
        df_conf,
        eligible_case_ids,
        samples_per_rule_per_class=SAMPLES_PER_RULE_PER_CLASS,
        seed=SEED,
        max_total_pairs=MAX_TOTAL_PAIRS,
        max_pairs_per_rule=MAX_PAIRS_PER_RULE,
    )
    print(f"Sampled (case, rule) pairs: {len(pairs)} (CAP={MAX_TOTAL_PAIRS})")
    print()

    # ---- LLM accounting ----
    print("> LLM evaluation")
    llm_queries_executed = 0
    total_tokens_used = 0

    rows: List[Dict[str, Any]] = []
    current_model: Optional[str] = None
    model_start_timer: Optional[float] = None
    model, settings = next(iter(llm_config.items()))
    for case_id, constraint, det_label in pairs:
        if model != current_model:
            if current_model is not None:
                model_minutes = (time.perf_counter() - model_start_timer) / 60.0
                print(f"Model completed: {current_model} ({model_minutes:.2f} minutes) \n")
            print(f"Will evaluate model: {model} ({settings['provider']})")
            current_model = model
            model_start_timer = time.perf_counter()

        client = clients[model]
        trace_df = traces.get(case_id)

        if trace_df is None:
            rows.append({
                "model": model,
                "prompt": args.prompt,
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

        prompt = build_llm_prompt(
            case_id,
            constraint,
            gold_meta,
            trace_text,
            prompt_key=args.prompt,
        )

        out, tokens_used = call_llm_json(
            client,
            prompt,
            model=model,
            settings=settings,
        )
        llm_queries_executed += 1
        total_tokens_used += int(tokens_used)

        llm_label = str(out.get("llm_label", "uncertain")).strip().lower()
        conf = float(out.get("confidence", 0.0)) if out.get("confidence") is not None else 0.0
        rationale = str(out.get("rationale", "")).strip()

        # Accept the former key as a compatibility fallback for model outputs
        # generated with an older prompt version.
        ev = out.get("evidence_lines", out.get("evidence_events", []))
        ev_str = json.dumps(ev if isinstance(ev, list) else [], ensure_ascii=False)

        rows.append({
            "model": model,
            "prompt": args.prompt,
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

    if current_model is not None:
        model_minutes = (time.perf_counter() - model_start_timer) / 60.0
        print(f"Model completed: {current_model} ({model_minutes:.2f} minutes)")

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

    df_res.to_csv(results_path, sep=CSV_SEP, index=False)
    summary_df.to_csv(summary_path, sep=CSV_SEP, index=False)
    per_rule_df.to_csv(per_rule_path, sep=CSV_SEP, index=False)
    mismatches.to_csv(mismatches_path, sep=CSV_SEP, index=False)

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
        "models": " | ".join(llm_config.keys()),
        "n_models": len(llm_config),
        "prompt": args.prompt,
    }])
    run_meta_df.to_csv(run_meta_path, sep=CSV_SEP, index=False)

    print("\n=== OUTPUT FILES ===")
    print(f"Saved results: {results_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved per-rule: {per_rule_path}")
    print(f"Saved mismatches: {mismatches_path}")
    print(f"Saved run metadata: {run_meta_path}")
    print(f"Saved rules used: {RULES_USED_PATH}")
    print()

    print("\n=== QUICK SUMMARY ===")
    print(summary_df.to_string(index=False))
    print("\n=== RUN METADATA ===")
    print(run_meta_df.to_string(index=False))

    program_end_dt = datetime.now().astimezone()
    total_seconds = time.perf_counter() - program_start_timer
    total_minutes, remaining_seconds = divmod(total_seconds, 60)
    print(f"\nProgramme finished: {program_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"Total time taken: {int(total_minutes)} minutes "
        f"and {remaining_seconds:.1f} seconds"
    )


if __name__ == "__main__":
    main()
