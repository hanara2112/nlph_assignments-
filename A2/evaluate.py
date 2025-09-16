#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import pandas as pd


def char_set(spans_df: pd.DataFrame) -> Dict[str, Set[Tuple[str, int]]]:
    # Since we have a single class ENT, keep per-concept id correctness check separate.
    # For IoU we collect characters for the ENT class only.
    ent_chars: Set[Tuple[str, int]] = set()
    for _, r in spans_df.iterrows():
        nid = str(r['note_id'])
        s, e = int(r['start']), int(r['end'])
        for pos in range(s, e):
            ent_chars.add((nid, pos))
    return {'ENT': ent_chars}


def macro_char_iou(pred: pd.DataFrame, gold: pd.DataFrame) -> float:
    P = char_set(pred)
    G = char_set(gold)
    classes = sorted(set(P.keys()) | set(G.keys()))
    ious = []
    for c in classes:
        pset = P.get(c, set())
        gset = G.get(c, set())
        inter = len(pset & gset)
        union = len(pset | gset)
        iou = (inter / union) if union else 0.0
        ious.append(iou)
    return sum(ious) / len(ious) if ious else 0.0


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description='Evaluate macro char-IoU for ENT spans')
    ap.add_argument('--pred', type=str, required=True)
    ap.add_argument('--gold', type=str, required=True)
    args = ap.parse_args(argv)

    pred = pd.read_csv(Path(args.pred))
    gold = pd.read_csv(Path(args.gold))
    # gold has concept_id; we only need spans here for IoU
    gold_spans = gold[['note_id', 'start', 'end']].copy()
    score = macro_char_iou(pred[['note_id', 'start', 'end']], gold_spans)
    print(f'Macro char-IoU (ENT): {score:.4f}')


if __name__ == '__main__':
    main()



