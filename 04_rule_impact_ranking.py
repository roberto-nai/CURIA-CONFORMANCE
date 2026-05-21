#!/usr/bin/env python3
"""
04_rule_impact_ranking.py

Rule impact ranking (leave-one-rule-out) for Declare conformance matrices.

Expected input format (Declare4Py export):
- semicolon-separated CSV
- header: constraint names (often with " | | |" noise)
- rows: 0/1 values (1 = violation)
- case id column may be absent

Outputs:
- conformance_impact/conformance_impact.csv
- conformance_impact/case_compliance_scores.csv
"""

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


# =============================
# CONFIGURATION
# =============================

CONFORMANCE_PATH = Path("conformance_results/conformance_results.csv")
GOLD_RULES_PATH = Path("event_rules/workflow_rules.csv")

OUTPUT_PATH = Path("conformance_impact/conformance_impact.csv")
CASE_SCORES_PATH = Path("conformance_impact/case_compliance_scores.csv")

VIOLATION_VALUE = 1  # 1 = violation, 0 = compliant
CSV_SEP = ";"  # centralised CSV separator


# =============================
# LOAD DATA
# =============================

def _clean_constraint_name(name: str) -> str:
    """
    Normalize Declare constraint column names exported by Declare4Py.

    Declare4Py may export headers such as
    `Precedence[Opinion, Judgment] | | |`; this function keeps only the left
    constraint part.

    Args:
        name: Raw column header.

    Returns:
        Clean constraint name, or empty string for null input.
    """
    if name is None:
        return ""
    return str(name).split("|", 1)[0].strip()


def load_conformance_matrix(path: Path) -> pd.DataFrame:
    """
    Load and normalize a conformance matrix from CSV.

    Processing steps include:
    - header trimming and constraint-name cleanup,
    - case-id column detection and index normalization,
    - numeric conversion of all rule columns,
    - dropping fully empty columns and filling missing values.

    Args:
        path: Path to conformance CSV.

    Returns:
        DataFrame indexed by case identifier with integer 0/1 rule values.
    """
    df = pd.read_csv(path, sep=CSV_SEP)
    df.columns = [c.strip() for c in df.columns]

    # Clean constraint header noise
    df = df.rename(columns={c: _clean_constraint_name(c) for c in df.columns})

    # If there is a case id column, set it as index; otherwise create a stable one
    case_id_col = None
    for cand in ["CaseID", "case:concept:name", "case_id", "case", "Case"]:
        if cand in df.columns:
            case_id_col = cand
            break

    if case_id_col is not None:
        df[case_id_col] = df[case_id_col].astype(str).str.strip()
        df = df.set_index(case_id_col)
    else:
        df.index = [f"case_{i:06d}" for i in range(1, len(df) + 1)]

    # Convert all columns to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop empty columns (can happen if there is a trailing ';' in export)
    df = df.dropna(axis=1, how="all").fillna(0).astype(int)

    print(f"Loaded conformance matrix with {len(df)} cases and {df.shape[1]} rules.")
    return df


def load_gold_rules(path: Path) -> pd.DataFrame:
    """
    Load GOLD rules table from CSV.

    Args:
        path: Path to workflow-rules CSV.

    Returns:
        DataFrame containing GOLD rule metadata.
    """
    df = pd.read_csv(path, sep=CSV_SEP)
    df.columns = df.columns.str.strip()
    return df


# =============================
# COMPLIANCE SCORE
# =============================

def compute_case_scores(df_constraints: pd.DataFrame) -> pd.Series:
    """
    Compute case-level compliance score in [0, 1].

    Formula:
        score(i) = 1 - (#violations(i) / #rules)

    Args:
        df_constraints: Rule matrix (rows=cases, columns=rules, 1=violation).

    Returns:
        Series indexed by case id with name `compliance_score`.
    """
    n_rules = df_constraints.shape[1]
    if n_rules == 0:
        return pd.Series(1.0, index=df_constraints.index, name="compliance_score")

    num_viol = (df_constraints == VIOLATION_VALUE).sum(axis=1)
    scores = 1.0 - (num_viol / n_rules)
    scores.name = "compliance_score"
    return scores


def compute_global_score(case_scores: pd.Series) -> float:
    """
    Compute the global compliance score as the mean of case scores.

    Args:
        case_scores: Case-level compliance scores.

    Returns:
        Mean compliance score across all cases.
    """
    return float(case_scores.mean())


# =============================
# RULE IMPACT RANKING
# =============================

def rule_impact_ranking(df_constraints: pd.DataFrame) -> pd.DataFrame:
    """
    Leave-one-rule-out impact on global compliance score.

    impact_delta = global_score_without_rule - baseline_global_score

    If impact_delta > 0:
      removing the rule increases compliance -> rule is "costly"/hard.

    Args:
        df_constraints: Rule matrix with binary violation values.

    Returns:
        DataFrame with one row per rule and impact metrics.
    """
    n_rules = df_constraints.shape[1]
    if n_rules == 0:
        return pd.DataFrame(columns=[
            "rule", "baseline_global_score", "global_score_without_rule",
            "impact_delta", "violation_rate", "num_violations", "num_cases"
        ])

    baseline_case_scores = compute_case_scores(df_constraints)
    baseline_global = compute_global_score(baseline_case_scores)
    n_cases = len(df_constraints)

    rows = []
    for rule_col in df_constraints.columns:
        df_wo = df_constraints.drop(columns=[rule_col])

        case_scores_wo = compute_case_scores(df_wo)
        global_wo = compute_global_score(case_scores_wo)

        violation_rate = float((df_constraints[rule_col] == VIOLATION_VALUE).mean())
        num_violations = int((df_constraints[rule_col] == VIOLATION_VALUE).sum())

        rows.append({
            "rule": rule_col,
            "baseline_global_score": baseline_global,
            "global_score_without_rule": global_wo,
            "impact_delta": global_wo - baseline_global,  # >0 => removing increases score
            "violation_rate": violation_rate,
            "num_violations": num_violations,
            "num_cases": n_cases,
        })

    impact_df = pd.DataFrame(rows).sort_values(
        by=["impact_delta", "violation_rate", "num_violations"],
        ascending=[False, False, False],
    )
    return impact_df


# =============================
# GOLD ENRICHMENT (best-effort)
# =============================

_CONSTRAINT_RE = re.compile(r"^(?P<tpl>[A-Za-z_]+)\[(?P<a>.*?),(?P<b>.*?)\]\s*$")


def parse_constraint(constraint_name: str):
    """
    Parse a binary Declare constraint into trigger and related activities.

    Args:
        constraint_name: Constraint string like `Response[A, B]`.

    Returns:
        Tuple `(A, B)` if parsing succeeds, otherwise `(None, None)`.
    """
    m = _CONSTRAINT_RE.match(constraint_name.strip())
    if not m:
        return None, None
    return m.group("a").strip(), m.group("b").strip()


def enrich_with_gold(impact_df: pd.DataFrame, gold_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich impact ranking rows with GOLD rule metadata when matching is possible.

    Matching is performed on `(trigger_concept, related_concept)` parsed from
    each constraint string.

    Args:
        impact_df: Rule-impact ranking output.
        gold_df: GOLD rules table with metadata columns.

    Returns:
        Enriched DataFrame. If required columns are missing or impact is empty,
        the original impact DataFrame is returned.
    """
    needed = {"rule_id", "name", "trigger_concept", "related_concept", "workflow_type", "optionality"}
    if not needed.issubset(set(gold_df.columns)) or impact_df.empty:
        return impact_df

    gold_df = gold_df.copy()
    gold_df["trigger_concept"] = gold_df["trigger_concept"].astype(str).str.strip()
    gold_df["related_concept"] = gold_df["related_concept"].astype(str).str.strip()

    enriched_rows = []
    for _, row in impact_df.iterrows():
        constraint = str(row["rule"])
        trigger, related = parse_constraint(constraint)

        out = row.to_dict()
        match = gold_df[
            (gold_df["trigger_concept"] == str(trigger).strip()) &
            (gold_df["related_concept"] == str(related).strip())
        ]

        if not match.empty:
            m = match.iloc[0]
            out.update({
                "gold_rule_id": str(m["rule_id"]),
                "gold_rule_name": str(m["name"]),
                "gold_workflow_type": str(m["workflow_type"]),
                "gold_optionality": str(m["optionality"]),
            })
        else:
            out.update({
                "gold_rule_id": "",
                "gold_rule_name": "",
                "gold_workflow_type": "",
                "gold_optionality": "",
            })

        enriched_rows.append(out)

    result = pd.DataFrame(enriched_rows)
    front = ["gold_rule_id", "gold_rule_name", "gold_workflow_type", "gold_optionality"]
    rest = [c for c in result.columns if c not in front]
    return result[front + rest]


# =============================
# MAIN
# =============================

def main() -> None:
    """
    Run the full rule-impact ranking workflow.

    Steps:
    1. Load conformance matrix.
    2. Compute and export case-level compliance scores.
    3. Compute leave-one-rule-out impact ranking.
    4. Load GOLD rules and enrich ranking metadata.
    5. Save final impact ranking CSV.
    """
    print("Loading conformance matrix...")
    df_conf = load_conformance_matrix(CONFORMANCE_PATH)

    # ---- NEW: export case-level compliance scores (per proceeding) ----
    print("Computing case-level compliance scores...")
    baseline_case_scores = compute_case_scores(df_conf)
    n_rules = df_conf.shape[1]

    case_scores_df = pd.DataFrame({
        "CaseID": df_conf.index.astype(str),
        "num_violations": (df_conf == VIOLATION_VALUE).sum(axis=1).astype(int),
        "n_rules": int(n_rules),
        "compliance_score": baseline_case_scores.astype(float),
    })

    CASE_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    case_scores_df.to_csv(CASE_SCORES_PATH, sep=CSV_SEP, index=False)
    print(f"Saved case-level scores to: {CASE_SCORES_PATH}")

    print("Computing rule impact ranking...")
    impact_df = rule_impact_ranking(df_conf)

    print("Loading GOLD rules...")
    gold_df = load_gold_rules(GOLD_RULES_PATH)

    print("Enriching with GOLD metadata...")
    impact_df = enrich_with_gold(impact_df, gold_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    impact_df.to_csv(OUTPUT_PATH, sep=CSV_SEP, index=False)

    print(f"Saved impact ranking to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()