#!/usr/bin/env python3
"""
LLM-only rule extraction (workflow rules) from a regulatory PDF.

Goal
----
PDF text  ->  LLM extracts workflow rules (JSON)  ->  Save JSON + CSV

This script DOES NOT check violations on the event log.

Dependencies
------------
pip install openai pypdf pandas python-dotenv

Environment
-----------
export OPENAI_API_KEY="..."

References (API usage)
----------------------
- OpenAI Responses API reference: https://platform.openai.com/docs/api-reference/responses
- openai-python usage examples: https://github.com/openai/openai-python
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pypdf import PdfReader

from openai import OpenAI

from dotenv import load_dotenv
import os

from datetime import datetime
import csv

# -----------------------------
# Run constants
# -----------------------------
PDF_DIR = "curia_texts"
PDF_FILE = "OJ_L_202402173_EN_TXT.pdf"
PDF_PATH = Path(PDF_DIR) / PDF_FILE
OUT_DIR = "curia_rules"
OUT_PATH = Path(OUT_DIR)
MODEL_NAME = "gpt-5.2"
NUM_RULES = 5
CSV_SEP = ";"  

# -----------------------------
# Schemas (v1)
# -----------------------------

RULES_SCHEMA_V1: Dict[str, Any] = {
    "type": "object",
    "required": ["ruleset_id", "source", "rules"],
    "properties": {
        "ruleset_id": {"type": "string"},
        "source": {
            "type": "object",
            "required": ["title", "doc_id"],
            "properties": {
                "title": {"type": "string"},
                "doc_id": {"type": "string"},
                "notes": {"type": "string"},
            },
        },
        "rules": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "rule_id",
                    "name",
                    "natural_language_rule",
                    "workflow_type",
                    "trigger_concept",
                    "related_concept",
                    "temporal_condition",
                    "optionality",
                    "checkability",
                    "excerpt_reference",
                ],
                "properties": {
                    "rule_id": {"type": "string"},
                    "name": {"type": "string"},
                    "natural_language_rule": {"type": "string"},
                    "workflow_type": {
                        "type": "string",
                        "enum": [
                            "PRECEDENCE",
                            "RESPONSE",
                            "EXISTENCE",
                            "ABSENCE",
                            "ABSENCE_AFTER",
                            "CONDITIONAL_PRECEDENCE",
                            "TIME_CONSTRAINT",
                        ],
                    },
                    "trigger_concept": {"type": "string"},
                    "related_concept": {"type": ["string", "null"]},
                    "temporal_condition": {"type": ["string", "null"]},
                    "optionality": {"type": "string", "enum": ["MANDATORY", "OPTIONAL", "CONDITIONAL"]},
                    "checkability": {"type": "string", "enum": ["full", "partial"]},
                    "assumptions": {"type": "array", "items": {"type": "string"}},
                    "excerpt_reference": {"type": "string"},
                },
            },
        },
    },
}


# -----------------------------
# Utilities
# -----------------------------

def read_pdf_text(pdf_path: Path) -> str:
    """
    Extract text from a PDF, preserving rough page boundaries.

    Each page is prefixed with a marker in the form `--- PAGE N ---` to keep
    traceability between extracted text and source location.

    Args:
        pdf_path: Path to the input PDF file.

    Returns:
        A non-empty string containing the full extracted text.

    Raises:
        ValueError: If no text can be extracted from the PDF.
    """
    reader = PdfReader(str(pdf_path))
    parts: List[str] = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        txt = txt.replace("\r\n", "\n").replace("\r", "\n")
        parts.append(f"\n\n--- PAGE {i+1} ---\n{txt}")
    text = "\n".join(parts).strip()
    if not text:
        raise ValueError("No text extracted from PDF (is it scanned/image-only?).")
    return text


def normalise_whitespace(text: str) -> str:
    """
    Apply light whitespace normalization to reduce prompt noise.

    The function collapses repeated spaces/tabs and reduces long blank-line
    sequences while preserving paragraph-level separation.

    Args:
        text: Raw text to normalize.

    Returns:
        Normalized text with trimmed leading/trailing whitespace.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Parse JSON from model output with a fallback extraction strategy.

    Parsing attempts, in order:
    1. Direct `json.loads(text)`
    2. Regex extraction of a JSON object block, then parsing

    Args:
        text: Raw model output that should contain a JSON object.

    Returns:
        Parsed JSON payload as a dictionary.

    Raises:
        ValueError: If no JSON object can be found in the text.
        json.JSONDecodeError: If a candidate JSON block is found but invalid.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{.*\}\s*$", text, flags=re.DOTALL)
    if not m:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not m:
        raise ValueError("Model output is not valid JSON and no JSON object could be extracted.")

    return json.loads(m.group(0))


def validate_minimal_structure(payload: Dict[str, Any]) -> None:
    """
    Validate the minimal expected structure of extracted rules payload.

    This is intentionally lightweight and avoids external `jsonschema`
    dependencies. It checks required top-level keys and required keys for each
    rule object.

    Args:
        payload: Parsed JSON payload returned by the model.

    Raises:
        ValueError: If required top-level keys are missing, `rules` is empty,
            or a required key is missing in any rule object.
    """
    for k in ("ruleset_id", "source", "rules"):
        if k not in payload:
            raise ValueError(f"Invalid output: missing top-level key '{k}'.")

    if not isinstance(payload["rules"], list) or len(payload["rules"]) == 0:
        raise ValueError("Invalid output: 'rules' must be a non-empty list.")

    for r in payload["rules"]:
        for k in (
            "rule_id",
            "name",
            "natural_language_rule",
            "workflow_type",
            "trigger_concept",
            "related_concept",
            "temporal_condition",
            "optionality",
            "checkability",
            "excerpt_reference",
        ):
            if k not in r:
                raise ValueError(f"Invalid rule object: missing key '{k}' in one rule.")


def rules_to_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Flatten a validated rules payload into a tabular DataFrame.

    Args:
        payload: Rules payload containing top-level metadata and `rules`.

    Returns:
        DataFrame with one row per rule and normalized columns ready for CSV
        export.
    """
    rows: List[Dict[str, Any]] = []
    for r in payload["rules"]:
        rows.append(
            {
                "ruleset_id": payload.get("ruleset_id", ""),
                "rule_id": r.get("rule_id", ""),
                "name": r.get("name", ""),
                "workflow_type": r.get("workflow_type", ""),
                "optionality": r.get("optionality", ""),
                "checkability": r.get("checkability", ""),
                "trigger_concept": r.get("trigger_concept", ""),
                "related_concept": r.get("related_concept", ""),
                "temporal_condition": r.get("temporal_condition", ""),
                "natural_language_rule": r.get("natural_language_rule", ""),
                "excerpt_reference": r.get("excerpt_reference", ""),
                "assumptions": " | ".join(r.get("assumptions", []) or []),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# Prompting
# -----------------------------

def build_extraction_prompt(reg_text: str, *, num_rules: int = 5) -> str:
    """
    Build the extraction prompt used by the LLM.

    The prompt enforces:
    - a fixed number of rules,
    - a closed set of allowed activity labels,
    - strict workflow constraint types,
    - JSON output compatible with the internal schema.

    Args:
        reg_text: Normalized regulatory text to analyze.
        num_rules: Exact number of rules the model must output.

    Returns:
        A fully formatted prompt string.
    """
    schema_str = json.dumps(RULES_SCHEMA_V1, ensure_ascii=False)

    return f"""
You are a legal process mining expert.

Your task is to extract EXACTLY {num_rules} workflow compliance rules
from the regulatory text.

CRITICAL CONSTRAINT:
You MUST use ONLY the following Activity labels as procedural concepts.
You are NOT allowed to introduce any other event types.

Allowed Activity labels (use exact strings):

1. Application (OJ)
2. Judgment
3. Judgment (OJ)
4. Request for a preliminary ruling
5. Order
6. Order (Information)
7. Abstract
8. Opinion
9. Statement of case or written observations
10. Judgment (Summary)
11. Removal (OJ)
12. Judgment (Information)
13. Statement of case or written observations - Corrigendum
14. Order (OJ)
15. Request for a preliminary ruling - Addendum
16. National decision following the preliminary ruling
17. Request for a preliminary ruling - Corrigendum
18. Order (Summary)
19. Judgment (extracts)

STRICT RULES:
- If a rule requires events not in this list, DO NOT extract it.
- Do NOT use concepts like service, defence, cross-appeal, objection, hearing closed, reopening.
- Focus only on ordering, existence, response, and absence constraints.
- Each rule must be atomic (one constraint only).
- Prefer rules that are fully checkable from timestamps and ordering.

Workflow types allowed:
PRECEDENCE
RESPONSE
EXISTENCE
ABSENCE
ABSENCE_AFTER
CONDITIONAL_PRECEDENCE

For each rule:
- rule_id
- name
- workflow_type
- optionality (MANDATORY or CONDITIONAL)
- trigger_concept (must match one of the allowed Activity labels)
- related_concept (or null if not applicable)
- temporal_condition ("before", "after", or null)
- natural_language_rule
- excerpt_reference (short quote from text, max 25 words)
- checkability = "full"
- assumptions (empty list unless strictly necessary)

OUTPUT JSON must follow this JSON Schema (v1):
{schema_str}

REGULATORY TEXT:
{reg_text}
""".strip()


# -----------------------------
# OpenAI call
# -----------------------------


def call_llm_extract_rules(
    prompt: str,
    *,
    model: str,
    temperature: float = 0.0,
    max_output_tokens: int = 1800,
) -> str:
    """
    Call the OpenAI Responses API to extract rules from the prompt.

    Environment variable loading is performed via `.env` and process env.

    Args:
        prompt: Full prompt sent to the model.
        model: Model identifier to use for inference.
        temperature: Sampling temperature for generation.
        max_output_tokens: Maximum number of output tokens.

    Returns:
        Raw text output returned by the API, stripped.

    Raises:
        EnvironmentError: If `OPENAI_API_KEY` is not available.
    """

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in environment or .env file")

    client = OpenAI(api_key=api_key)

    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    return resp.output_text.strip()


# -----------------------------
# Main
# -----------------------------

def run(
    pdf_path: Path,
    out_dir: Path,
    *,
    model: str,
    num_rules: int = 5,
    csv_sep: str = CSV_SEP
) -> Tuple[Path, Path]:
    """
    Execute the end-to-end rule extraction pipeline.

    Steps:
    1. Ensure output directory exists.
    2. Read and normalize PDF text.
    3. Build extraction prompt and call the LLM.
    4. Parse and minimally validate JSON payload.
    5. Save JSON, CSV, and prompt artifacts.

    Args:
        pdf_path: Path to the source regulatory PDF.
        out_dir: Directory where output artifacts are written.
        model: Model identifier used for extraction.
        num_rules: Number of rules requested from the model.
        csv_sep: CSV separator used when exporting tabular rules.

    Returns:
        Tuple containing paths to the generated JSON and CSV files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_text = read_pdf_text(pdf_path)
    reg_text = normalise_whitespace(raw_text)

    prompt = build_extraction_prompt(reg_text, num_rules=num_rules)

    llm_out = call_llm_extract_rules(prompt, model=model)

    payload = safe_json_loads(llm_out)
    validate_minimal_structure(payload)

    # Save JSON
    json_path = out_dir / "workflow_rules.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save CSV
    df_rules = rules_to_dataframe(payload)
    csv_path = out_dir / "workflow_rules.csv"
    df_rules.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL, sep=csv_sep)

    # Also save the prompt for reproducibility
    prompt_path = out_dir / "prompt_used.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    return json_path, csv_path


if __name__ == "__main__":
    print()
    print("*** PROGRAM START ***")
    print()
    time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Time start: {time_stamp}")
    print()
    
    print(f"1) Processing PDF: {PDF_PATH}")
    json_path, csv_path = run(
        pdf_path=PDF_PATH,
        out_dir=OUT_PATH,
        model=MODEL_NAME,
        num_rules=NUM_RULES,
        csv_sep = CSV_SEP
    )
    print(f"2) Saved JSON: {json_path}")
    print(f"3) Saved CSV:  {csv_path}")
    print()
    time_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Time end: {time_end}")
    print("*** PROGRAM END ***")
    print()