#!/usr/bin/env python3
import argparse
import json
import os
import re
import string
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Paths and dataset defaults
# -----------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / 'Assignment 2 Dataset'
DEFAULT_OUT_DIR = PROJECT_DIR / 'outputs'
DEFAULT_RESOURCES_DIR = PROJECT_DIR / 'resources'


# -----------------------------
# Utilities
# -----------------------------

Token = Tuple[str, int, int]

_token_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_whitespace_re = re.compile(r"\s+")
_punct_tbl = str.maketrans('', '', string.punctuation)


def tokenize_with_offsets(text: str) -> List[Token]:
    tokens: List[Token] = []
    for m in _token_re.finditer(text):
        tok = m.group(0)
        tokens.append((tok, m.start(), m.end()))
    return tokens


def align_bio(tokens: List[Token], spans: List[Tuple[int, int]]) -> List[str]:
    labels: List[str] = []
    for i, (_, s, e) in enumerate(tokens):
        lab = 'O'
        for (ss, ee) in spans:
            if s >= ss and e <= ee:
                prev_inside = i > 0 and (tokens[i - 1][1] >= ss and tokens[i - 1][2] <= ee)
                lab = 'B-ENT' if not prev_inside else 'I-ENT'
                break
        labels.append(lab)
    return labels


def bio_to_spans(tokens: List[Token], labels: Sequence[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    prev_end: Optional[int] = None
    for i, (_, s, e) in enumerate(tokens):
        lab = labels[i]
        if lab.startswith('B-'):
            if start is not None and prev_end is not None:
                spans.append((start, prev_end))
            start = s
        elif lab == 'O':
            if start is not None and prev_end is not None:
                spans.append((start, prev_end))
                start = None
        prev_end = e
    if start is not None and prev_end is not None:
        spans.append((start, prev_end))
    return spans


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(_punct_tbl)
    text = _whitespace_re.sub(' ', text).strip()
    return text


def tokenize_ws(text: str) -> List[str]:
    return [t for t in _whitespace_re.split(text) if t]


# -----------------------------
# CRF NER
# -----------------------------


def maybe_install_crfSuite() -> None:
    try:
        import sklearn_crfsuite  # noqa: F401
        from seqeval.metrics import classification_report, f1_score  # noqa: F401
    except Exception:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sklearn-crfsuite', 'seqeval'])


def word2features(sent_tokens: List[Token], i: int) -> Dict[str, object]:
    tok, _, _ = sent_tokens[i]
    features: Dict[str, object] = {
        'bias': 1.0,
        'word.lower()': tok.lower(),
        'word.isupper()': tok.isupper(),
        'word.istitle()': tok.istitle(),
        'word.isdigit()': tok.isdigit(),
        'suffix3': tok[-3:],
        'suffix2': tok[-2:],
        'prefix1': tok[:1],
        'prefix2': tok[:2],
        'prefix3': tok[:3],
    }
    if i > 0:
        ptok = sent_tokens[i - 1][0]
        features.update({
            '-1:word.lower()': ptok.lower(),
            '-1:word.istitle()': ptok.istitle(),
            '-1:word.isupper()': ptok.isupper(),
        })
    else:
        features['BOS'] = True
    if i < len(sent_tokens) - 1:
        ntok = sent_tokens[i + 1][0]
        features.update({
            '+1:word.lower()': ntok.lower(),
            '+1:word.istitle()': ntok.istitle(),
            '+1:word.isupper()': ntok.isupper(),
        })
    else:
        features['EOS'] = True
    return features


def prepare_xy(seqs: List[Dict[str, object]]) -> Tuple[List[List[Dict[str, object]]], List[List[str]]]:
    X: List[List[Dict[str, object]]] = []
    y: List[List[str]] = []
    for seq in seqs:
        sent_tokens: List[Token] = seq['tokens']  # type: ignore[index]
        labels: List[str] = seq['labels']  # type: ignore[index]
        X.append([word2features(sent_tokens, i) for i in range(len(sent_tokens))])
        y.append(labels)
    return X, y


def build_sequences(notes_df: pd.DataFrame, spans_map: Dict[str, List[Tuple[int, int]]]) -> List[Dict[str, object]]:
    sequences: List[Dict[str, object]] = []
    for _, row in notes_df.iterrows():
        nid = str(row['note_id'])
        text = str(row['text'])
        toks = tokenize_with_offsets(text)
        spans = spans_map.get(nid, [])
        labels = align_bio(toks, spans)
        sequences.append({'note_id': nid, 'tokens': toks, 'labels': labels})
    return sequences


def train_and_predict_crf(train_notes: pd.DataFrame,
                          train_ann: pd.DataFrame,
                          test_notes: pd.DataFrame,
                          test_ann: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[float]]:
    maybe_install_crfSuite()
    from sklearn_crfsuite import CRF
    try:
        from seqeval.metrics import f1_score
    except Exception:
        f1_score = None  # type: ignore[assignment]

    train_spans_by_note: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for _, r in train_ann.iterrows():
        nid = str(r['note_id'])
        train_spans_by_note[nid].append((int(r['start']), int(r['end'])))

    test_spans_by_note: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    if test_ann is not None:
        for _, r in test_ann.iterrows():
            nid = str(r['note_id'])
            test_spans_by_note[nid].append((int(r['start']), int(r['end'])))

    train_seqs = build_sequences(train_notes, train_spans_by_note)
    test_seqs = build_sequences(test_notes, test_spans_by_note)

    X_train, y_train = prepare_xy(train_seqs)
    X_test, y_test = prepare_xy(test_seqs)

    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)

    y_pred = crf.predict(X_test)
    f1 = None
    if test_ann is not None and f1_score is not None:
        try:
            f1 = float(f1_score(y_test, y_pred))  # type: ignore[arg-type]
        except Exception:
            f1 = None

    # Convert predictions to span rows
    pred_rows: List[Dict[str, object]] = []
    for seq, labs in zip(test_seqs, y_pred):
        nid = seq['note_id']  # type: ignore[index]
        toks: List[Token] = seq['tokens']  # type: ignore[index]
        spans = bio_to_spans(toks, labs)
        for (s, e) in spans:
            pred_rows.append({'note_id': nid, 'start': s, 'end': e, 'label': 'ENT'})

    pred_df = pd.DataFrame(pred_rows)
    return pred_df, f1


def build_surface_lexicon(train_notes: pd.DataFrame, train_ann: pd.DataFrame) -> Dict[str, Counter]:
    notes_text_map = train_notes.set_index('note_id')['text'].to_dict()
    surface_to_concepts: Dict[str, Counter] = defaultdict(Counter)
    for _, r in train_ann.iterrows():
        nid = r['note_id']
        s, e = int(r['start']), int(r['end'])
        text = notes_text_map.get(nid, '')
        mention = text[s:e].strip()
        if mention:
            surface_to_concepts[mention.lower()][str(r['concept_id'])] += 1
    return surface_to_concepts


def attach_concepts_via_lexicon(pred_df: pd.DataFrame,
                                test_notes: pd.DataFrame,
                                surface_to_concepts: Dict[str, Counter]) -> pd.DataFrame:
    out = pred_df.copy()
    out['concept_id'] = None
    notes_text_map_test = test_notes.set_index('note_id')['text'].to_dict()
    for i, row in out.iterrows():
        nid = row['note_id']
        s, e = int(row['start']), int(row['end'])
        text = notes_text_map_test.get(nid, '')
        mention = text[s:e].strip().lower()
        if mention in surface_to_concepts:
            best_cid, _ = surface_to_concepts[mention].most_common(1)[0]
            out.at[i, 'concept_id'] = best_cid
    return out


# -----------------------------
# Linker (candidate gen + selection)
# -----------------------------


def load_snomed_dict_or_fallback(snomed_csv: Path,
                                 train_notes: pd.DataFrame,
                                 train_ann: pd.DataFrame) -> pd.DataFrame:
    if snomed_csv.exists():
        df = pd.read_csv(snomed_csv)
        df['synonyms'] = df.get('synonyms', '').fillna('').astype(str)
        rows: List[Dict[str, str]] = []
        for _, r in df.iterrows():
            cid = str(r['snomed_id'])
            term = str(r['term'])
            rows.append({'snomed_id': cid, 'term': term})
            syns = [s.strip() for s in r['synonyms'].split('|') if s.strip()]
            for s in syns:
                rows.append({'snomed_id': cid, 'term': s})
        dict_df = pd.DataFrame(rows)
        if not dict_df.empty:
            return dict_df
    # Fallback to surface lexicon
    notes_map = train_notes.set_index('note_id')['text'].to_dict()
    rows_lex: List[Dict[str, str]] = []
    for _, r in train_ann.iterrows():
        nid = r['note_id']
        s, e = int(r['start']), int(r['end'])
        mention = str(notes_map.get(nid, ''))[s:e].strip()
        if mention:
            rows_lex.append({'snomed_id': str(r['concept_id']), 'term': mention})
    dict_df = pd.DataFrame(rows_lex).drop_duplicates()
    return dict_df


def build_exact_index(snomed_terms: pd.DataFrame) -> Dict[str, List[Tuple[str, str]]]:
    snomed_terms = snomed_terms.copy()
    snomed_terms['norm_term'] = snomed_terms['term'].astype(str).map(normalize)
    snomed_terms = snomed_terms.drop_duplicates(['snomed_id', 'norm_term']).reset_index(drop=True)
    exact_index: Dict[str, List[Tuple[str, str]]] = {}
    for _, r in snomed_terms.iterrows():
        exact_index.setdefault(r['norm_term'], []).append((str(r['snomed_id']), str(r['term'])))
    return exact_index


def augment_snomed_terms_simple_rules(snomed_terms: pd.DataFrame) -> pd.DataFrame:
    """Add simple synonym variants: common abbreviations and basic de-hyphenization.

    This helps experiment 4.1 (dictionary-based linking with simple linguistic rules).
    """
    if snomed_terms.empty:
        return snomed_terms
    rows: List[Dict[str, str]] = snomed_terms[['snomed_id', 'term']].astype(str).to_dict('records')
    abbr_map: Dict[str, List[str]] = {
        'diabetes mellitus': ['dm'],
        'hypertension': ['htn'],
        'myocardial infarction': ['mi'],
        'coronary artery disease': ['cad'],
        'chronic kidney disease': ['ckd'],
        'atrial fibrillation': ['af'],
        'chronic obstructive pulmonary disease': ['copd'],
        'blood pressure': ['bp'],
        'congestive heart failure': ['chf'],
        'cerebrovascular accident': ['cva'],
    }
    for _, r in snomed_terms.iterrows():
        cid = str(r['snomed_id'])
        term = str(r['term'])
        tl = term.lower()
        # Abbreviations if phrase is contained
        for phrase, abbrs in abbr_map.items():
            if phrase in tl:
                for ab in abbrs:
                    rows.append({'snomed_id': cid, 'term': ab})
        # De-hyphenize/cleanup variants
        if '-' in term:
            rows.append({'snomed_id': cid, 'term': term.replace('-', ' ')})
        if '/' in term:
            rows.append({'snomed_id': cid, 'term': term.replace('/', ' ')})
        tclean = term.replace('.', '').strip()
        if tclean and tclean != term:
            rows.append({'snomed_id': cid, 'term': tclean})
    out = pd.DataFrame(rows).drop_duplicates()
    return out


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    den = len(sa | sb)
    return (len(sa & sb) / den) if den else 0.0


def levenshtein_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + cost,
            )
            prev = cur
    dist = dp[m]
    return 1.0 - (dist / max(n, m))


def generate_candidates(span_text: str,
                        snomed_terms: pd.DataFrame,
                        exact_index: Dict[str, List[Tuple[str, str]]],
                        top_n: int = 5) -> List[Dict[str, object]]:
    norm = normalize(span_text)
    if norm in exact_index:
        return [{'snomed_id': cid, 'term': term, 'score': 1.0} for cid, term in exact_index[norm]][:top_n]
    toks_q = tokenize_ws(norm)
    rows: List[Dict[str, object]] = []
    # Pre-normalized terms not required here; compute on the fly
    for _, r in snomed_terms.iterrows():
        t = normalize(str(r['term']))
        lev = levenshtein_sim(norm, t)
        jac = jaccard(toks_q, tokenize_ws(t))
        score = 0.6 * lev + 0.4 * jac
        if score > 0:
            rows.append({'snomed_id': str(r['snomed_id']), 'term': str(r['term']), 'score': float(score)})
    rows.sort(key=lambda x: -x['score'])
    return rows[:top_n]


def generate_candidates_similarity_only(span_text: str,
                                       snomed_terms: pd.DataFrame,
                                       top_n: int = 5) -> List[Dict[str, object]]:
    norm = normalize(span_text)
    toks_q = tokenize_ws(norm)
    rows: List[Dict[str, object]] = []
    for _, r in snomed_terms.iterrows():
        t = normalize(str(r['term']))
        lev = levenshtein_sim(norm, t)
        jac = jaccard(toks_q, tokenize_ws(t))
        score = 0.6 * lev + 0.4 * jac
        if score > 0:
            rows.append({'snomed_id': str(r['snomed_id']), 'term': str(r['term']), 'score': float(score)})
    rows.sort(key=lambda x: -x['score'])
    return rows[:top_n]


def generate_candidates_exact_only(span_text: str,
                                   exact_index: Dict[str, List[Tuple[str, str]]],
                                   top_n: int = 5) -> List[Dict[str, object]]:
    norm = normalize(span_text)
    if norm in exact_index:
        return [{'snomed_id': cid, 'term': term, 'score': 1.0} for cid, term in exact_index[norm]][:top_n]
    return []


def select_best(candidates: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (-float(x['score']), str(x['snomed_id'])))[0]


# -----------------------------
# LLM + FAISS (optional)
# -----------------------------


def ensure_faiss_resources(snomed_terms: pd.DataFrame,
                           embed_model_name: str):
    """Build in-memory FAISS index and embedder. Returns (index, embedder, terms_list).

    We keep this in-memory to avoid adding extra files; re-use across calls within the same run.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError('Missing optional deps for LLM/FAISS: install sentence-transformers and faiss-cpu') from e

    embedder = SentenceTransformer(embed_model_name)
    term_texts: List[str] = snomed_terms['term'].astype(str).tolist()
    vectors = embedder.encode(term_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    dim = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    # Keep minimal meta alongside
    terms_list: List[Tuple[str, str]] = list(zip(snomed_terms['snomed_id'].astype(str).tolist(), term_texts))
    return index, embedder, terms_list


def retrieve_candidates_faiss(span_text: str,
                              index,
                              embedder,
                              terms_list: List[Tuple[str, str]],
                              top_n: int = 5) -> List[Dict[str, object]]:
    qvec = embedder.encode([span_text], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(qvec, top_n)
    out: List[Dict[str, object]] = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        cid, term = terms_list[int(idx)]
        out.append({'snomed_id': cid, 'term': term, 'score': float(score)})
    return out


def load_llm_pipeline(base_model: str,
                      lora_adapter: Optional[str] = None,
                      device: str = 'cpu'):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        has_peft = True
        try:
            from peft import PeftModel  # type: ignore
        except Exception:
            has_peft = False
    except Exception as e:
        raise RuntimeError('Missing optional deps for LLM: install transformers, torch, and optionally peft') from e

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    if lora_adapter:
        if not has_peft:
            raise RuntimeError('peft is required to use a LoRA adapter')
        from peft import PeftModel  # type: ignore
        model = PeftModel.from_pretrained(model, lora_adapter)
    if device != 'cpu' and torch.cuda.is_available():
        model = model.to('cuda')
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if (device != 'cpu' and torch.cuda.is_available()) else -1)
    return pipe


def llm_choose_concept(pipe,
                       span_text: str,
                       candidates: List[Dict[str, object]],
                       max_new_tokens: int = 64) -> Optional[Dict[str, object]]:
    if not candidates:
        return None
    # Short, deterministic prompt; ask for just the ID.
    options = []
    for i, c in enumerate(candidates, 1):
        options.append(f"{i}. {c['term']} [id={c['snomed_id']}]")
    prompt = (
        "You are a clinical concept linker. Given an entity mention and a list of SNOMED CT candidate concepts, "
        "return ONLY the chosen id.\n" \
        f"Mention: {span_text}\n" \
        "Candidates:\n" + "\n".join(options) + "\nAnswer with only the id:"
    )
    try:
        gen = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        text = gen[0]['generated_text']
    except Exception:
        return sorted(candidates, key=lambda x: -float(x['score']))[0]
    # Extract the first id substring present in candidates
    cand_ids = {str(c['snomed_id']): c for c in candidates}
    for cid in cand_ids.keys():
        if cid in text:
            return cand_ids[cid]
    # Fallback to best score
    return sorted(candidates, key=lambda x: -float(x['score']))[0]


def extract_spans_with_llm(pipe, notes_df: pd.DataFrame, max_new_tokens: int = 256) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in notes_df.iterrows():
        nid = str(row['note_id'])
        text = str(row['text'])
        instr = (
            "Extract clinical entity spans from the note. Return ONLY valid JSON list of objects with 'start' and 'end' (character offsets).\n"
            "Example: [{\"start\": 10, \"end\": 18}]\n"
            f"Note:\n{text}\n"
            "JSON: "
        )
        try:
            gen = pipe(instr, max_new_tokens=max_new_tokens, do_sample=False)
            out = gen[0]['generated_text']
            # Try to locate a JSON array
            lb = out.find('[')
            rb = out.rfind(']')
            arr_text = out[lb:rb + 1] if (lb != -1 and rb != -1 and rb > lb) else out
            spans = json.loads(arr_text)
            for sp in spans:
                s = int(sp['start'])
                e = int(sp['end'])
                if 0 <= s < e <= len(text):
                    rows.append({'note_id': nid, 'start': s, 'end': e, 'label': 'ENT'})
        except Exception:
            # If LLM fails, skip this note
            continue
    return pd.DataFrame(rows)


# -----------------------------
# CLI commands
# -----------------------------


def cmd_ner(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_notes = pd.read_csv(data_dir / 'training-notes.csv')
    train_ann = pd.read_csv(data_dir / 'train-annotations.csv')
    test_notes = pd.read_csv(data_dir / 'test-notes.csv')
    test_ann_path = data_dir / 'test-annotations.csv'
    test_ann = pd.read_csv(test_ann_path) if test_ann_path.exists() else None

    if getattr(args, 'ner_backend', 'crf') == 'llm':
        pipe = load_llm_pipeline(args.llm_base_model, getattr(args, 'lora_adapter', None), getattr(args, 'device', 'cpu'))
        preds_df = extract_spans_with_llm(pipe, test_notes, max_new_tokens=getattr(args, 'max_new_tokens', 256))
        f1 = None
    else:
        preds_df, f1 = train_and_predict_crf(train_notes, train_ann, test_notes, test_ann)

    ner_csv = out_dir / 'ner_predictions.csv'
    preds_df.to_csv(ner_csv, index=False)

    if args.attach_concept:
        surface_lex = build_surface_lexicon(train_notes, train_ann)
        preds_full = attach_concepts_via_lexicon(preds_df, test_notes, surface_lex)
        full_csv = out_dir / 'ner_predictions_with_concept.csv'
        preds_full.to_csv(full_csv, index=False)

    if f1 is not None:
        print(f'NER test F1 (seqeval micro): {f1:.4f}')
    print(f'Saved NER spans to: {ner_csv}')


def cmd_link(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_path = out_dir / 'ner_predictions.csv'
    if not preds_path.exists():
        print(f'ERROR: Missing predicted spans at {preds_path}. Run ner first.', file=sys.stderr)
        sys.exit(1)

    train_notes = pd.read_csv(data_dir / 'training-notes.csv')
    train_ann = pd.read_csv(data_dir / 'train-annotations.csv')
    test_notes = pd.read_csv(data_dir / 'test-notes.csv')
    test_ann_path = data_dir / 'test-annotations.csv'
    test_ann = pd.read_csv(test_ann_path) if test_ann_path.exists() else None

    preds = pd.read_csv(preds_path)

    default_snomed = DEFAULT_RESOURCES_DIR / 'snomed_dictionary.csv'
    snomed_csv = Path(args.snomed_csv) if args.snomed_csv else default_snomed
    snomed_terms = load_snomed_dict_or_fallback(snomed_csv, train_notes, train_ann)
    # Apply simple dictionary augmentations for experiment 4.1
    snomed_terms = augment_snomed_terms_simple_rules(snomed_terms)
    exact_index = build_exact_index(snomed_terms)

    notes_map_test = test_notes.set_index('note_id')['text'].to_dict()

    cand_records: List[Dict[str, object]] = []
    link_mode = getattr(args, 'link_mode', 'hybrid')
    if link_mode == 'llm':
        # Build FAISS resources once
        index, embedder, terms_list = ensure_faiss_resources(snomed_terms, getattr(args, 'embed_model', 'sentence-transformers/all-MiniLM-L6-v2'))
        pipe = load_llm_pipeline(args.llm_base_model, getattr(args, 'lora_adapter', None), getattr(args, 'device', 'cpu'))
        for _, r in preds.iterrows():
            nid = str(r['note_id'])
            s, e = int(r['start']), int(r['end'])
            span = str(notes_map_test.get(nid, ''))[s:e]
            cands = retrieve_candidates_faiss(span, index, embedder, terms_list, top_n=args.top_n)
            cand_records.append({'note_id': nid, 'start': s, 'end': e, 'span': span, 'candidates': cands})
    else:
        for _, r in preds.iterrows():
            nid = str(r['note_id'])
            s, e = int(r['start']), int(r['end'])
            span = str(notes_map_test.get(nid, ''))[s:e]
            if link_mode == 'dict':
                cands = generate_candidates_exact_only(span, exact_index, top_n=args.top_n)
            elif link_mode == 'sim':
                cands = generate_candidates_similarity_only(span, snomed_terms, top_n=args.top_n)
            else:  # hybrid
                cands = generate_candidates(span, snomed_terms, exact_index, top_n=args.top_n)
            cand_records.append({'note_id': nid, 'start': s, 'end': e, 'span': span, 'candidates': cands})

    cand_path = out_dir / 'linker_candidates.jsonl'
    with open(cand_path, 'w') as f:
        for rec in cand_records:
            f.write(json.dumps(rec) + '\n')

    linked_rows: List[Dict[str, object]] = []
    if link_mode == 'llm':
        pipe = load_llm_pipeline(args.llm_base_model, getattr(args, 'lora_adapter', None), getattr(args, 'device', 'cpu'))
        for rec in cand_records:
            best = llm_choose_concept(pipe, rec['span'], rec['candidates'], max_new_tokens=getattr(args, 'max_new_tokens', 64))
            linked_rows.append({
                'note_id': rec['note_id'],
                'start': rec['start'],
                'end': rec['end'],
                'span': rec['span'],
                'snomed_id': best['snomed_id'] if best else None,
                'matched_term': best['term'] if best else None,
                'score': best['score'] if best else None,
            })
    else:
        for rec in cand_records:
            best = select_best(rec['candidates'])
            linked_rows.append({
                'note_id': rec['note_id'],
                'start': rec['start'],
                'end': rec['end'],
                'span': rec['span'],
                'snomed_id': best['snomed_id'] if best else None,
                'matched_term': best['term'] if best else None,
                'score': best['score'] if best else None,
            })

    linked_df = pd.DataFrame(linked_rows)
    link_path = out_dir / 'linker_links.csv'
    linked_df.to_csv(link_path, index=False)

    # Optional evaluation if test annotations are available
    if test_ann is not None:
        gold_map: Dict[Tuple[str, int, int], str] = {}
        for _, rr in test_ann.iterrows():
            gold_map[(str(rr['note_id']), int(rr['start']), int(rr['end']))] = str(rr['concept_id'])
        num = 0
        correct = 0
        for _, rr in linked_df.iterrows():
            key = (str(rr['note_id']), int(rr['start']), int(rr['end']))
            if key in gold_map:
                num += 1
                pred_cid = None if pd.isna(rr['snomed_id']) else str(rr['snomed_id'])
                if pred_cid == gold_map[key]:
                    correct += 1
        acc = (correct / num) if num else 0.0
        print(f'Exact-span linking accuracy: {acc:.4f} (N={num})')

    print(f'Saved candidates to: {cand_path}')
    print(f'Saved links to: {link_path}')


def build_llm_training_pairs(train_notes: pd.DataFrame, train_ann: pd.DataFrame, max_examples: Optional[int] = None) -> List[Dict[str, str]]:
    """Create instruction→target pairs for span extraction SFT.

    Target is a JSON array of {start,end} spans for the given note.
    """
    notes_map = train_notes.set_index('note_id')['text'].to_dict()
    spans_by_note: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for _, r in train_ann.iterrows():
        spans_by_note[str(r['note_id'])].append((int(r['start']), int(r['end'])))

    pairs: List[Dict[str, str]] = []
    for nid, text in notes_map.items():
        spans = [{'start': s, 'end': e} for (s, e) in sorted(spans_by_note.get(str(nid), []))]
        instr = (
            "Extract clinical entity spans from the note. Return ONLY JSON list of {start,end}.\n"
            "Example: [{\"start\": 10, \"end\": 18}]\n"
            f"Note:\n{text}\n"
            "JSON: "
        )
        target = json.dumps(spans)
        pairs.append({'prompt': instr, 'target': target})
        if max_examples is not None and len(pairs) >= max_examples:
            break
    return pairs


def cmd_llm_train(args: argparse.Namespace) -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        raise RuntimeError('Install transformers, torch, peft to use llm-train') from e

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    lora_out = out_dir / 'lora_adapter'
    lora_out.mkdir(parents=True, exist_ok=True)

    train_notes = pd.read_csv(data_dir / 'training-notes.csv')
    train_ann = pd.read_csv(data_dir / 'train-annotations.csv')
    pairs = build_llm_training_pairs(train_notes, train_ann, max_examples=getattr(args, 'max_examples', None))

    tokenizer = AutoTokenizer.from_pretrained(args.llm_base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, examples: List[Dict[str, str]]):
            self.examples = examples
        def __len__(self) -> int:
            return len(self.examples)
        def __getitem__(self, idx: int):
            ex = self.examples[idx]
            text = ex['prompt'] + ex['target']
            enc = tokenizer(text, truncation=True, max_length=args.max_seq_len, padding='max_length', return_tensors='pt')
            input_ids = enc['input_ids'][0]
            attn = enc['attention_mask'][0]
            labels = input_ids.clone()
            return {'input_ids': input_ids, 'attention_mask': attn, 'labels': labels}

    ds = TextDataset(pairs)

    model = AutoModelForCausalLM.from_pretrained(args.llm_base_model)
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=getattr(args, 'lora_target_modules', ['q_proj', 'v_proj']), lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lora_cfg)

    args_tr = TrainingArguments(
        output_dir=str(out_dir / 'llm_train_runs'),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        fp16=False,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=1,
        report_to=[],
        remove_unused_columns=False,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=args_tr, train_dataset=ds, data_collator=collator)
    trainer.train()
    model.save_pretrained(str(lora_out))
    print(f'Saved LoRA adapter to: {lora_out}')


def cmd_all(args: argparse.Namespace) -> None:
    # Run NER then linker
    cmd_ner(args)
    cmd_link(args)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='A2 Baseline: NER (CRF) + SNOMED Linker')
    p.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR), help='Dataset directory containing notes/annotations CSVs')
    p.add_argument('--out-dir', type=str, default=str(DEFAULT_OUT_DIR), help='Output directory for predictions and linker files')
    sub = p.add_subparsers(dest='cmd', required=False)

    p.add_argument('--attach-concept', action='store_true', help='During NER, also attach concept_id by surface lexicon')

    # Linker-specific flags
    p.add_argument('--snomed-csv', type=str, default='', help='Optional SNOMED dictionary CSV path (snomed_id,term,synonyms)')
    p.add_argument('--top-n', type=int, default=5, help='Number of candidates to retain per span')
    p.add_argument('--link-mode', type=str, choices=['dict', 'sim', 'hybrid', 'llm'], default='hybrid', help='Dictionary, similarity, hybrid, or LLM-based selection')
    p.add_argument('--embed-model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='Embedding model for FAISS retrieval (LLM mode)')
    p.add_argument('--llm-base-model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2', help='Base HF model for LLM modes')
    p.add_argument('--lora-adapter', type=str, default='', help='Optional LoRA adapter path for the LLM')
    p.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    p.add_argument('--max-new-tokens', type=int, default=64, help='Generation tokens for LLM prompts')
    p.add_argument('--ner-backend', type=str, choices=['crf', 'llm'], default='crf', help='Use CRF (default) or LLM for span extraction')

    # LLM training (LoRA) flags
    p.add_argument('--num-epochs', type=int, default=1, help='LLM SFT epochs')
    p.add_argument('--batch-size', type=int, default=1, help='LLM SFT batch size')
    p.add_argument('--lr', type=float, default=5e-5, help='LLM SFT learning rate')
    p.add_argument('--max-seq-len', type=int, default=1024, help='Max sequence length for SFT')
    p.add_argument('--lora-r', type=int, default=8, help='LoRA rank')
    p.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha')
    p.add_argument('--max-examples', type=int, default=200, help='Max training examples for quick SFT')

    # Commands
    sub_ner = sub.add_parser('ner', help='Run NER only')
    sub_ner.add_argument('--attach-concept', action='store_true', help='Also attach concept_id by surface lexicon')
    sub_link = sub.add_parser('link', help='Run linker only (expects ner_predictions.csv in out-dir)')
    sub_all = sub.add_parser('all', help='Run both NER and linker')
    sub_llmtrain = sub.add_parser('llm-train', help='Fine-tune LLM with LoRA for span extraction (quick SFT)')

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # normalize args in case subparser was used
    if not getattr(args, 'data_dir', None):
        setattr(args, 'data_dir', str(DEFAULT_DATA_DIR))
    if not getattr(args, 'out_dir', None):
        setattr(args, 'out_dir', str(PROJECT_DIR))

    cmd = args.cmd or 'all'
    if cmd == 'ner':
        cmd_ner(args)
    elif cmd == 'link':
        cmd_link(args)
    elif cmd == 'llm-train':
        cmd_llm_train(args)
    else:
        cmd_all(args)


if __name__ == '__main__':
    main()


