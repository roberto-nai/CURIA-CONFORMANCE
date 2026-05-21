#!/usr/bin/env python3
"""
03_declare_model.py

Generate an executable Declare model (.decl) from GOLD workflow rules (JSON),
with a validation + normalisation layer that makes the model "engine-ready"
for Declare4Py.

Methodology-friendly design
---------------------------
1) Load GOLD rules (workflow rules in JSON).
2) Map each rule to a Declare template (rule → template).
3) Normalise activity names (optional aliasing for LTL / safety).
4) Ensure template executability in Declare4Py:
   - Apply deterministic fallbacks for unsupported templates.
   - Record every action in a validation report.
5) Write:
   - curia_model.decl (executable model)
   - declare_model_build_report.csv (traceable build/validation report)
   - declare_model_build_report.json (same content, structured)
   - activity_alias_map.json (only if aliasing is enabled)

Notes
-----
- In Declare4Py, some templates are not implemented (e.g., NotSuccession).
  We therefore operationalise ABSENCE_AFTER rules as NotResponse[A, B] by default,
  which is executable and captures the intended closure semantics.

Requirements
------------
pip install pandas
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ===============================
# CONFIG
# ===============================

CSV_SEP = ";"

GOLD_RULES_PATH = Path("event_rules/workflow_rules.json")
OUTPUT_MODEL_PATH = Path("declare_model/curia_model.decl")

OUT_DIR = OUTPUT_MODEL_PATH.parent
REPORT_CSV_PATH = OUT_DIR / "declare_model_build_report.csv"
REPORT_JSON_PATH = OUT_DIR / "declare_model_build_report.json"

# If True, activities are converted to safe aliases for LTL / parsers.
# For your current pipeline we keep it OFF, because we avoid LTL here.
ENABLE_ACTIVITY_ALIASING = False
ALIAS_MAP_PATH = OUT_DIR / "activity_alias_map.json"


# ===============================
# Data structures
# ===============================

@dataclass
class BuildRow:
    rule_id: str
    name: str
    workflow_type: str
    trigger_concept: str
    related_concept: str
    declare_line: str
    status: str
    action_taken: str
    notes: str


# ===============================
# LOAD GOLD RULES
# ===============================

def load_gold_rules(path: Path) -> List[Dict[str, Any]]:
    """
    Load workflow rules from JSON file.

    Supports both accepted input shapes:
    - object with key `rules` containing a list
    - bare list of rule objects

    Args:
        path: Path to the JSON rules file.

    Returns:
        List of rule dictionaries.

    Raises:
        ValueError: If the parsed structure is not a list of rules.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rules = payload.get("rules", payload)  # allow both {"rules":[...]} and bare list
    if not isinstance(rules, list):
        raise ValueError("workflow_rules.json must contain a list of rules or a dict with key 'rules'.")
    return rules


# ===============================
# Activity aliasing (optional)
# ===============================

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_]+")


def _to_safe_alias(name: str) -> str:
    """
    Convert an activity label into a parser-safe alias.

    The alias keeps alphanumeric and underscore characters, collapsing any
    invalid sequence into a single underscore.

    Args:
        name: Original activity label.

    Returns:
        Safe alias string, or `ACT` if the normalized value is empty.
    """
    alias = _SAFE_NAME_RE.sub("_", name.strip())
    alias = re.sub(r"_+", "_", alias).strip("_")
    return alias or "ACT"


def build_activity_alias_map(rules: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Build alias mapping for all activities referenced in rules.

    Args:
        rules: List of workflow rule dictionaries.

    Returns:
        Dictionary mapping original activity labels to safe aliases.
    """
    activities: List[str] = []
    for r in rules:
        a = str(r.get("trigger_concept", "")).strip()
        b = str(r.get("related_concept", "")).strip() if r.get("related_concept") is not None else ""
        if a:
            activities.append(a)
        if b:
            activities.append(b)

    unique = sorted(set(activities))
    alias_map = {act: _to_safe_alias(act) for act in unique}
    return alias_map


def maybe_alias(act: str, alias_map: Dict[str, str]) -> str:
    """
    Return aliased activity name when available.

    Args:
        act: Activity label to resolve.
        alias_map: Mapping from original labels to aliases.

    Returns:
        Aliased label if present in map, otherwise the original label.
    """
    return alias_map.get(act, act)


# ===============================
# DECLARE TEMPLATE MAPPING (engine-ready)
# ===============================

def rule_to_declare_engine_ready(
    rule: Dict[str, Any],
    *,
    alias_map: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], str, str]:
    """
    Map one workflow rule to an executable Declare template line.

    The function applies deterministic fallbacks for unsupported or fragile
    cases so the generated model remains executable in Declare4Py.

    Args:
        rule: Workflow rule dictionary.
        alias_map: Optional activity alias map used for safe labels.

    Returns:
        Tuple `(declare_line, action_taken, notes)` where `declare_line` can be
        `None` if the rule is skipped.

    action_taken examples:
    - "mapped"
    - "fallback"
    - "skipped"
    """
    alias_map = alias_map or {}

    rtype = str(rule.get("workflow_type", "")).strip()
    A_raw = str(rule.get("trigger_concept", "")).strip()
    B_raw = str(rule.get("related_concept", "")).strip() if rule.get("related_concept") is not None else ""

    if not A_raw:
        return None, "skipped", "Missing trigger_concept"

    A = maybe_alias(A_raw, alias_map)
    B = maybe_alias(B_raw, alias_map) if B_raw else ""

    # ---- Core mappings ----
    if rtype in {"PRECEDENCE", "CONDITIONAL_PRECEDENCE"}:
        if not B:
            return None, "skipped", "Precedence requires related_concept"
        return f"Precedence[{A}, {B}]", "mapped", "Rule mapped to Precedence"

    if rtype == "RESPONSE":
        if not B:
            return None, "skipped", "Response requires related_concept"
        return f"Response[{A}, {B}]", "mapped", "Rule mapped to Response"

    if rtype == "EXISTENCE":
        return f"Existence[{A}]", "mapped", "Rule mapped to Existence"

    # ---- ABSENCE_AFTER: operationalise to executable template ----
    # Instead of LTL[...] (fragile + spaces), use NotResponse[A,B]
    # which is supported by Declare4Py and matches the intended closure semantics:
    # "After A occurs, B must not occur afterwards."
    if rtype == "ABSENCE_AFTER":
        if not B:
            return None, "skipped", "Absence-after requires related_concept"
        return f"NotResponse[{A}, {B}]", "fallback", "ABSENCE_AFTER operationalised as NotResponse (Declare4Py executable)"

    # Unknown type
    return None, "skipped", f"Unsupported workflow_type='{rtype}'"


# ===============================
# BUILD DECLARE MODEL + REPORT
# ===============================

def build_declare_model_and_report(
    rules: List[Dict[str, Any]],
    *,
    enable_aliasing: bool,
) -> Tuple[str, List[BuildRow], Optional[Dict[str, str]]]:
    """
    Build executable Declare model text and validation report rows.

    Args:
        rules: Input workflow rules.
        enable_aliasing: Whether to convert activity labels to safe aliases.

    Returns:
        Tuple `(model_text, report_rows, alias_map)`.
        `alias_map` is `None` when aliasing is disabled.
    """
    alias_map = build_activity_alias_map(rules) if enable_aliasing else None

    lines: List[str] = []
    lines.append("# CURIA DECLARE MODEL (engine-ready)")
    lines.append("# Generated from GOLD rules with validation + fallbacks for Declare4Py")
    lines.append("")

    report_rows: List[BuildRow] = []

    for r in rules:
        rule_id = str(r.get("rule_id", "")).strip()
        name = str(r.get("name", "")).strip()
        wtype = str(r.get("workflow_type", "")).strip()
        A = str(r.get("trigger_concept", "")).strip()
        B = str(r.get("related_concept", "")).strip() if r.get("related_concept") is not None else ""

        declare_line, action_taken, notes = rule_to_declare_engine_ready(r, alias_map=alias_map)

        if declare_line:
            lines.append(declare_line)
            status = "included"
        else:
            status = "excluded"

        report_rows.append(
            BuildRow(
                rule_id=rule_id,
                name=name,
                workflow_type=wtype,
                trigger_concept=A,
                related_concept=B,
                declare_line=declare_line or "",
                status=status,
                action_taken=action_taken,
                notes=notes,
            )
        )

    model_text = "\n".join(lines).strip() + "\n"
    return model_text, report_rows, alias_map


def save_report(report_rows: List[BuildRow]) -> None:
    """
    Save build report in CSV and JSON formats.

    Args:
        report_rows: Validation/build report rows generated during model build.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([r.__dict__ for r in report_rows])

    df.to_csv(REPORT_CSV_PATH, sep=CSV_SEP, index=False)
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in report_rows], f, ensure_ascii=False, indent=2)


# ===============================
# MAIN
# ===============================

def main() -> None:
    """Run the full Declare model generation pipeline from configured paths."""
    rules = load_gold_rules(GOLD_RULES_PATH)

    model_text, report_rows, alias_map = build_declare_model_and_report(
        rules,
        enable_aliasing=ENABLE_ACTIVITY_ALIASING,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_MODEL_PATH, "w", encoding="utf-8") as f:
        f.write(model_text)

    save_report(report_rows)

    if ENABLE_ACTIVITY_ALIASING and alias_map is not None:
        with open(ALIAS_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(alias_map, f, ensure_ascii=False, indent=2)

    included = sum(1 for r in report_rows if r.status == "included")
    excluded = sum(1 for r in report_rows if r.status == "excluded")

    print(f"Declare model saved to: {OUTPUT_MODEL_PATH.resolve()}")
    print(f"Report CSV saved to:   {REPORT_CSV_PATH.resolve()}")
    print(f"Report JSON saved to:  {REPORT_JSON_PATH.resolve()}")
    if ENABLE_ACTIVITY_ALIASING:
        print(f"Alias map saved to:    {ALIAS_MAP_PATH.resolve()}")
    print(f"Rules included: {included} | excluded: {excluded}")


if __name__ == "__main__":
    main()