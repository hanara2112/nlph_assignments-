### Clinical De-identification Report (Assignment 1 — Q2.2)

This report documents three approaches for de-identifying Protected Health Information (PHI) in clinical discharge reports and analyzes their behavior on the provided dataset. It is intended to be read together with the code and generated outputs under `Assignments/part2/`.

Dataset and EDA snapshot
- Source: `Discharge Reports Dataset.csv` (200 discharge summaries).
- Key fields de-identified: the free-text `report` column.
- Basic EDA (on all 200 notes):
  - Average length: ~10.9K characters (~1.7K words) per note (long-form).
  - PHI prevalence by simple patterns (share of notes containing): dates ≈ 100%, phone-like ≈ 49.5%, ID-like ≈ 98.5%, emails ≈ 11.5%.
  - Implication: A strong regex safety net is essential; any ML method should be paired with rule-based coverage for structured identifiers.

Outputs produced
- JSONL per method (one `{"text": "..."}` object per note):
  - `output/deid_method1_spacy_regex.jsonl`
  - `output/deid_method2_medner_regex.jsonl`
  - `output/deid_method3_llm.jsonl`
- Comparison/diagnostics:
  - `output/Deid Comparison Preview.csv` (side-by-side original vs. three methods)
  - `output/Deid Placeholder Counts.csv` (placeholder counts by note and method)
  - `output/Data Smell Counts.csv` (EDA counts per note)

---

### Method 1 — General NER (spaCy) + Regex Rules

Pipeline
- Model: `en_core_web_lg` (spaCy large English NER) for entity spans.
- Label mapping: PERSON → `[NAME]`; ORG → `[ORG]`; geographic types (GPE/LOC/FAC) → `[LOCATION]`.
- Regex pass (independent of NER) for structured identifiers:
  - Dates: multiple formats (ISO `YYYY-MM-DD`, `MM/DD/YYYY`, `DD-MM-YYYY`, and month-name patterns)
  - Emails, phone numbers, ID-like tokens (e.g., `MRN|ID|Unit No|Account` followed by digits)
- Span resolution: sort by `start`, prefer longer spans first; merge non-overlapping; replace with placeholders.
- Guardrails added in this submission:
  - Header/section filters to avoid masking section labels (e.g., “Discharge Date:”, “Attending:”) and other common headings.

Where it works well
- Dates/emails/phones/IDs are reliably masked via regex regardless of NER.
- PERSON/ORG/LOCATION detection is reasonable for common names and organizations mentioned in narrative text.
- Deterministic, fast, and easy to inspect or adjust with new patterns.

Where it fails (and why)
- Recall limits for clinical PHI: general-domain NER may miss hospital names, provider roles, and institution-specific phrasing (domain gap).
- False positives near headings or abbreviations (mitigated here with header filters but still possible on some abbreviations).
- Limited granularity: all locations become `[LOCATION]` (no subtype like `[CITY]`, `[HOSPITAL]`), which is acceptable for de-id but reduces analytic detail.

Takeaway
- As a baseline, M1 + regex is strong on structured PHI and reasonable on names/locations, but it can under-mask clinical-specific identifiers and still needs guardrails to avoid over-masking non-PHI tokens.

---

### Method 2 — Clinical/PHI NER (Transformers) + Regex

Pipeline
- We attempt a PHI-oriented token classification model first, then fall back if unavailable:
  1) `obi/deid_roberta_i2b2` (RoBERTa-based model trained/fine-tuned on i2b2 de-identification data) — preferred
  2) `d4data/biomedical-ner-all` — biomedical NER fallback
  3) `dslim/bert-base-NER` — general-domain fallback (least preferred)
- Inference: Hugging Face `pipeline("token-classification", aggregation_strategy="simple")` for span grouping.
- Label normalization: map provider/person-like labels to `PERSON`; organization/hospital labels to `[ORG]/[HOSPITAL]`; geographic labels to `[LOCATION]`; and retain `DATE/PHONE/EMAIL/ID`.
- Regex safety pass: same as M1 for dates/emails/phones/IDs.
- Header overlap filter: suppress predictions that overlap section-header prefixes (e.g., avoid turning “Discharge Date:” into `[ORG] Date:`).

Model details
- Architecture: transformer encoder (e.g., RoBERTa/BERT) with a token-classification head predicting BIO/IOB2 tags; we aggregate to contiguous spans.
- Likely training data (for PHI de-id checkpoints): i2b2 2014/2016 de-identification challenges and similar clinical corpora annotated for PHI categories. See the i2b2 NLP challenges (refer to the official i2b2 resources at [Partners i2b2](https://www.i2b2.org/NLP/)).
- Inference on long notes: handled in one pass by the pipeline; for very long inputs, chunking is often required, but here tokenization typically fits via internal truncation unless explicitly chunked.

Feasibility of training such models
- Data: requires substantial annotated PHI spans (dozens of entity types across thousands of notes). Annotation is expensive and requires strict privacy controls.
- Compute: fine-tuning a RoBERTa/BERT de-id model is feasible on a single modern GPU (e.g., 1× A100/3090) with a few GPU-hours to a couple GPU-days depending on data size and hyperparameters.
- Practical considerations:
  - Domain shift: if target notes differ from training data (hospital-specific templates), performance drops without further adaptation.
  - Privacy: training must occur in a secure environment; raw PHI must not leave controlled infrastructure.

Possible improvements / alternatives
- Domain adaptation: further fine-tune on a small, locally-annotated sample; perform continual learning with strict access controls.
- Span post-processing: unify near-duplicate spans, enforce header-safe replacements, and add regex cleanup for structured leaks.
- Ensembling: union spans from multiple NER models (e.g., PHI model + biomedical NER) plus regex to maximize recall; optionally add voting to reduce false positives.
- Gazetteers/dictionaries: hospital lists, provider rosters, and common clinic names to augment recall for local entities.

Observed behavior
- Compared to M1, M2 better recognizes clinical roles, hospitals, and biomedical name patterns. With header filtering, it avoids the most common heading mislabels. Residual misses or mislabels remain when only general-domain fallback loads.

---

### Method 3 — Instruction-Following LLM (Chunked) + Safety Regex

Pipeline
- Model: instruction-tuned causal LLM (default `microsoft/phi-3-mini-4k-instruct`; fallback to a small chat LLM if unavailable).
- Prompting: strict system prompt asking to replace PHI with placeholders and return only de-identified text.
- Long notes: chunked into ~2000-character segments with ~200-character overlaps to keep within the context window, then reassembled.
- Safety pass: a regex sweep after generation to enforce masking of dates/emails/phones/IDs if the LLM missed any.

Analysis vs. Methods 1–2
- Strengths:
  - Can mask subtle PHI that pattern-based or narrow NER might miss (e.g., implicit hospital mentions, nuanced person references) when the prompt is followed.
  - Flexible and easy to iterate via prompt engineering.
- Weaknesses (unmitigated):
  - Instruction drift: may echo the prompt, produce summaries, or fail to consistently use placeholders without strict post-processing.
  - Context window: long notes require chunking; naive concatenation causes lost context or truncation.
  - Determinism: without constraints, outputs can vary and introduce artifacts.
- With our mitigations:
  - Chunking + safety regex reduces leakage but does not guarantee perfect adherence to placeholder formats for categories beyond dates/phones/emails/IDs (e.g., person/place). A hybrid approach is recommended.

How to improve M3 further
- Constrained decoding: restrict the output vocabulary to placeholder tokens for detected spans (requires span candidates from NER/regex and a masking protocol).
- Two-stage approach: (1) LLM extracts PHI spans in a structured format (JSON), (2) a deterministic masker replaces text with placeholders.
- Few-shot prompting: include short, diverse examples showing precise placeholder usage.
- Stronger safety pass: expand regexes (international numbers; broader date formats) and add dictionary checks for facility names and provider titles.

Bottom line
- A pure-LLM approach is not yet reliable enough alone; the best performance is achieved by pairing LLM output with deterministic masking or by using an ensemble that unions spans from M1/M2 with LLM suggestions and then applies a strict replacement step.

---

### Comparative Summary and Recommendations

Coverage and precision (qualitative)
- Structured PHI (dates/phones/emails/IDs): regex dominates; M1/M2/LLM benefit from the same safety pass. Residual leakage should be close to 0 with robust patterns.
- Persons/Providers/Hospitals/Locations:
  - M2 ≥ M1 on clinical entities (when a clinical PHI model is loaded). M1 can lag on provider/hospital recall.
  - M3 can be competitive, but only with chunking + strong post-processing; otherwise, it is inconsistent.

Over-masking (false positives)
- M1 may over-mask if unchecked (section labels, acronyms). Our header filter substantially reduces this.
- M2 may still confuse some headings or lab tokens if the fallback is general-domain; header overlap filtering mitigates common cases.
- M3 may paraphrase or summarize if unconstrained; our prompt and post-processing reduce but do not eliminate this risk.

Maintainability and runtime
- M1: fastest and simplest; easy to extend with regex/gazetteers.
- M2: strong balance of accuracy and determinism; needs model availability and occasional domain tuning.
- M3: heaviest and least deterministic; best used as a complementary pass with deterministic enforcement.

Recommended default for submission and deployment
- Use M2 + regex as the primary output when a clinical PHI model is available.
- Fall back to M1 + regex if M2 loads only a general model or infra is constrained.
- Optionally run M3 for additional suggestions but finalize with a deterministic replacement step (union of spans), followed by a robust safety regex sweep.

---

### References and Notes
- i2b2 de-identification challenges: see the official i2b2 NLP resources at [Partners i2b2](https://www.i2b2.org/NLP/).
- spaCy models: `en_core_web_lg` for general NER.
- Hugging Face models: `obi/deid_roberta_i2b2` (PHI de-id), `d4data/biomedical-ner-all` (biomedical NER), `dslim/bert-base-NER` (general NER).
- All methods here apply a shared span-merging policy and a regex safety pass for structured identifiers.


