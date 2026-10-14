# -*- coding: utf-8 -*-
"""Q43-only runner: byte-identical code path to run_causal.main()'s Q43 loop.
Runs the team's own run_day/solve_opt unchanged; only skips the serial Q42 pass."""
import sys, json, argparse
from pathlib import Path
import numpy as np

SRC = r'C:\Users\ahdezneytiry-boop1\.zcode\workspace\default\q4run\源代码'
sys.path.insert(0, SRC)
import data_core as dc
from run_causal import run_day, day_baselines, serial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attachments', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--initial-load-kw', type=float, required=True)
    parser.add_argument('--initial-price', type=float, required=True)
    parser.add_argument('--days', type=int, default=365)
    parser.add_argument('--settlement', choices=['refund', 'strict'], default='refund')
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()
    dc.ATT = str(Path(args.attachments).resolve())
    load, pv, dates = dc.load_attachment2()
    price, pdates = dc.load_attachment4()
    if not dates.equals(pdates):
        raise ValueError('Attachment date mismatch')
    fc = dc.load_attachment3()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    platform = []
    records = []
    energy = dc.E0_INIT
    for d in range(min(args.days, len(dates))):
        day = dates.iloc[d]
        lb, pb = day_baselines(load[:d], price[:d], dates.iloc[:d], day,
                               args.initial_load_kw, args.initial_price)
        fday = {h: dc.get_forecast_series(fc, day, h) for h in [0, 6, 12, 18]}
        v = run_day(load[d], price[d], pv[d], fday, lb, pb, energy, True,
                    args.settlement, args.alpha, platform, day)
        energy = v['E_end']
        v['date'] = str(day.date())
        records.append(v)
        print('Q43', v['date'], round(v['total'], 2), flush=True)
    (out / 'Q43.json').write_text(json.dumps(records, ensure_ascii=False, default=serial),
                                  encoding='utf-8')
    print('Q43-only done, days:', len(records))


if __name__ == '__main__':
    main()
