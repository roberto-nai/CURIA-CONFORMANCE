# CURIA Conformance Pipeline

Pipeline to extract procedural rules from CURIA regulatory text, transform them into a Declare model, run conformance checking on event logs, and produce impact/disagreement analyses with an LLM.

Regulatory reference:
https://eur-lex.europa.eu/IT/legal-content/summary/rules-of-procedure-of-the-court-of-justice-of-the-european-union.html

## Requirements

- Python 3.12 recommended
- Dependencies listed in `requirements.txt`
- API credentials for the configured remote LLM providers
- A local Ollama installation when an Ollama model is enabled

Quick setup:

```bash
python -m venv venv312
source venv312/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root and add only the credentials required
by the providers enabled in `llm_config.json`:

```dotenv
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Required only for a Meta model served by an OpenAI-compatible endpoint
META_API_KEY=your_meta_key
META_BASE_URL=https://your-endpoint.example/v1

# Optional; this is the default Ollama endpoint
OLLAMA_BASE_URL=http://localhost:11434/v1
```

Do not commit `.env` or expose API keys in source files.

## Pipeline

<img src="pipeline.png" alt="CURIA conformance pipeline" width="900">

<em>Overview of the end-to-end CURIA conformance analysis pipeline.</em>

## Script Execution Order

Recommended end-to-end order:

1. `01_llm_text_to_rules.py`  
   Extracts workflow rules (JSON/CSV) from the regulatory PDF using an LLM.
   
   Input data: CURIA regulatory text source (PDF) from `curia_texts/` and prompt/configuration parameters in the script.
   
   Output data: rule extraction artifacts in `curia_rules/`, including `workflow_rules.json`, `workflow_rules.csv`, and extraction prompt trace.

2. `02_declare_model.py`  
   Converts rules into an executable Declare model and generates build/validation reports.
   
   Input data: extracted rule set from `curia_rules/workflow_rules.json` or `curia_rules/workflow_rules.csv`.
  
   Output data: Declare model file and validation/build diagnostics in `declare_model/` (for example `curia_model.decl` and build report files).

3. `03_run_declare_conformance.py`  
   Runs Declare conformance checking on the event log and saves the matrix plus statistics.
  
   Input data: executable Declare model from `declare_model/` and event log data from `event_log/` (`.csv`/`.xes`).
  
   Output data: case-rule conformance matrix and aggregated statistics in `conformance_results/`.

4. `04_rule_impact_ranking.py`  
   Computes leave-one-rule-out impact per rule and produces ranking plus case-level scores.
   
   Input data: full conformance outputs from `conformance_results/`, especially per-case/per-rule compliance information.
   
   Output data: rule impact ranking and case-level compliance score files in `conformance_impact/`.

5. `05_llm_disagreement.py`  
   Compares the judgement of one model selected from `llm_config.json` with
   deterministic Declare labels on a balanced subset of case/rule pairs. It
   selects the top-impact rules and samples compliant and violating cases.
   
   Input data: `conformance_results/conformance_results.csv`, `conformance_impact/conformance_impact.csv`,
   `curia_rules/workflow_rules.csv`, `event_log/curia_log_en.csv`, `llm_config.json` and `llm_prompts.json`.
   
   Output data: row-level results, overall and per-rule summaries, mismatches,
   rules used, and run metadata in `llm_disagreement/`. The console also shows
   the programme start and finish times and execution duration.

   Run the script from the project root using:

   ```bash
   python3 05_llm_disagreement.py MODEL [PROMPT]
   ```

   `MODEL` is required and must exactly match a key in `llm_config.json`.
   `PROMPT` is optional and must match a key in `llm_prompts.json`; when it is
   omitted, the script uses `prompt_1`.

   Examples:

   ```bash
   # Use the default prompt_1
   python3 05_llm_disagreement.py gpt-4.1-2025-04-14

   # Select the prompt explicitly
   python3 05_llm_disagreement.py claude-sonnet-4-5-20250929 prompt_1

   # Run the local Ollama model
   python3 05_llm_disagreement.py llama3.2 prompt_1
   ```

   The program stops before making an LLM request if the model or prompt is not
   configured. Every CSV filename ends with the selected model name, for
   example `llm_disagreement_results_gpt-4_1-2025-04-14.csv`. Characters that
   are unsuitable for filenames are replaced with `_`, so `llama3.2` becomes
   `llama3_2`. CSV files are written for the current execution rather than
   appended. Each result row also records the selected model and prompt.

## LLM Configuration

`llm_config.json` is keyed by model identifier. Each model entry specifies its
provider and generation settings:

```json
{
  "model-name": {
    "provider": "OpenAI",
    "release_date": "YYYY-MM-DD",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_output_tokens": 256,
    "response_format": "JSON",
    "tools": "disabled"
  }
}
```

Supported providers are `OpenAI`, `Anthropic`, `Ollama`, and `Meta` through an
OpenAI-compatible endpoint. For Anthropic, the script sends `temperature` only
and leaves `top_p` at the model default because some Anthropic models reject
requests containing both parameters. OpenAI, Ollama, and Meta receive both
sampling parameters.

For a local Ollama model, ensure that Ollama is running and that the configured
model is available before starting the disagreement study.

Optional notebook:

- `00_log_analyser.ipynb`  
  Exploratory log analysis and diagnostics support (not required for the batch pipeline).

## Folder Organisation

Main logical structure:

- `curia_texts/`  
  Regulatory input (source PDF).

- `event_log/`  
  Event log inputs/intermediates (`.csv`, `.xes`).

- `curia_rules/`
  Extracted rules (JSON/CSV) and extraction prompt.

- `declare_model/`  
  Declare model `.decl` and build/validation reports.

- `conformance_results/`  
  Case/rule conformance outputs and aggregated statistics.

- `conformance_impact/`  
  Rule-impact ranking and case-level compliance scores.

- `llm_disagreement/`  
  Outputs for LLM vs deterministic conformance comparison.

## Key Outputs

- `curia_rules/workflow_rules.json` and `curia_rules/workflow_rules.csv`
- `declare_model/curia_model.decl`
- `conformance_results/conformance_results.csv`
- `conformance_impact/conformance_impact.csv`
- `llm_disagreement/llm_disagreement_results_<model>.csv`
- `llm_disagreement/llm_disagreement_summary_<model>.csv`
- `llm_disagreement/llm_disagreement_per_rule_<model>.csv`
- `llm_disagreement/llm_disagreement_mismatches_<model>.csv`
- `llm_disagreement/llm_disagreement_run_metadata_<model>.csv`
- `llm_disagreement/rules_used.txt`

## Operational Notes

- Scripts are configured via constants in each file's `CONFIGURATION`/`CONFIG` section.
- Model/provider settings for `05_llm_disagreement.py` are stored in `llm_config.json`.
- For reproducible runs, verify seeds and input/output paths before execution.

## Contact

For questions or collaboration requests, please contact: roberto.nai@unito.it
