#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description='Print a few notes and annotations')
    ap.add_argument('--data-dir', type=str, required=True)
    ap.add_argument('--n', type=int, default=2)
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    notes = pd.read_csv(data_dir / 'training-notes.csv')
    ann = pd.read_csv(data_dir / 'train-annotations.csv')

    for i, row in notes.head(args.n).iterrows():
        nid = row['note_id']
        text = str(row['text'])
        spans = ann[ann['note_id'] == nid].head(10)
        print('=' * 80)
        print(f'note_id: {nid}')
        print(text[:800].replace('\n', ' '))
        print('- spans: (showing up to 10)')
        for _, r in spans.iterrows():
            s, e = int(r['start']), int(r['end'])
            print(f'  [{s},{e}) -> {text[s:e].strip()} :: concept_id={r["concept_id"]}')


if __name__ == '__main__':
    main()


