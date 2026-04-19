#!/usr/bin/env python3
"""Repair ERROR rows in api_behavioral_sweep.json by replaying only failed requests."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper2_fidelity_calibrated.run_api_behavioral_sweep import LANGUAGES, build_prompt, api_call, norm
DEFAULT_API_KEY_PATHS = [
    Path.cwd() / 'api.txt',
    ROOT / 'api.txt',
    ROOT / 'paper2_fidelity_calibrated' / 'api.txt',
]


def resolve_api_key(cli_key: str) -> str:
    if cli_key:
        return cli_key.strip()
    for env_name in ('GOOGLE_API_KEY', 'GEMINI_API_KEY'):
        val = os.environ.get(env_name, '').strip()
        if val:
            return val
    for path in DEFAULT_API_KEY_PATHS:
        if not path.exists():
            continue
        text = path.read_text(encoding='utf-8').strip()
        if not text:
            continue
        if '=' in text:
            for line in text.splitlines():
                if '=' not in line:
                    continue
                k, v = line.split('=', 1)
                if k.strip() in {'GOOGLE_API_KEY', 'GEMINI_API_KEY'} and v.strip():
                    return v.strip().strip('"').strip("'")
        else:
            return text.strip().strip('"').strip("'")
    return ''


def replay_prediction(model_name: str, lang: str, src: str, n_icl: int, api_key: str, max_retries: int, retry_sleep: float) -> str:
    cfg = LANGUAGES[lang]
    prompt = build_prompt(cfg, src, n_icl)
    full_model = f'models/{model_name}' if not model_name.startswith('models/') else model_name
    last_pred = ''
    for attempt in range(max_retries + 1):
        pred = norm(api_call(full_model, prompt, api_key))
        last_pred = pred
        if pred and not pred.startswith('ERROR:'):
            return pred
        if attempt < max_retries:
            time.sleep(retry_sleep * (2 ** attempt))
    return last_pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=ROOT / 'paper2_fidelity_calibrated' / 'results' / 'api_behavioral_sweep.json')
    ap.add_argument('--output', type=Path, default=ROOT / 'paper2_fidelity_calibrated' / 'results' / 'api_behavioral_sweep_repaired.json')
    ap.add_argument('--api-key', default='')
    ap.add_argument('--max-retries', type=int, default=6)
    ap.add_argument('--retry-sleep', type=float, default=1.5)
    ap.add_argument('--request-delay', type=float, default=0.4)
    args = ap.parse_args()

    api_key = resolve_api_key(args.api_key)
    if not api_key:
        raise SystemExit('Missing API key: set GOOGLE_API_KEY / GEMINI_API_KEY or place api.txt at repo root')

    payload = json.loads(args.input.read_text(encoding='utf-8'))
    repaired = 0
    still_failed = 0

    for idx, row in enumerate(payload.get('items', []), start=1):
        pred = str(row.get('pred', ''))
        if not pred.startswith('ERROR:'):
            continue
        new_pred = replay_prediction(
            model_name=row['model'],
            lang=row['lang'],
            src=row['src'],
            n_icl=int(row['n_icl']),
            api_key=api_key,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
        )
        row['pred'] = new_pred
        gold_n = norm(row.get('gold', ''))
        row['em'] = float(new_pred == gold_n)
        row['fe'] = float(gold_n != '' and new_pred.startswith(gold_n[0])) if gold_n else 0.0
        if new_pred.startswith('ERROR:'):
            still_failed += 1
            status = 'FAILED'
        else:
            repaired += 1
            status = 'OK'
        print(f"[repair_api_behavioral] {idx}: {row['model']} {row['lang']} n_icl={row['n_icl']} {row['src']} -> {new_pred} [{status}]", flush=True)
        time.sleep(args.request_delay)

    from collections import defaultdict
    groups = defaultdict(list)
    for r in payload['items']:
        groups[(r['model'], r['lang'], r['n_icl'])].append(r)
    summary = []
    for (m, l, n), rs in sorted(groups.items()):
        summary.append({
            'model': m,
            'language': l,
            'n_icl': n,
            'n': len(rs),
            'exact_match': sum(float(r['em']) for r in rs) / len(rs),
            'first_entry': sum(float(r['fe']) for r in rs) / len(rs),
        })
    payload['summary'] = summary
    payload['repair_metadata'] = {
        'repaired_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'input_path': str(args.input),
        'repaired_count': repaired,
        'still_failed_count': still_failed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Saved repaired sweep: {args.output}")
    print(f"Repaired rows: {repaired}")
    print(f"Still failed: {still_failed}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
