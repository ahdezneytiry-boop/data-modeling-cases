# -*- coding: utf-8 -*-
"""
Q4 全年结果独立审计（与运行代码相互独立的核验实现）
- 不导入 run_causal/optimizer，用独立公式重算全部费用
- 检查 SOC 跨日连续、物理边界、充放互斥、供需平衡
- 检查评估窗口、跨口径方向性、与旧口径量级对比
"""
import json, os
import numpy as np
import pandas as pd

BASE = r'C:\Users\ahdezneytiry-boop1\.zcode\workspace\default\q4run'
ATT = os.path.join(BASE, '附件')
T = 144

def load_actual():
    ld = pd.read_excel(os.path.join(ATT, '附件2.xlsx'), '小区负载').iloc[:, 1:].values.astype(float)
    pv = pd.read_excel(os.path.join(ATT, '附件2.xlsx'), '光伏发电实际功率').iloc[:, 1:].values.astype(float)
    pr = pd.read_excel(os.path.join(ATT, '附件4.xlsx')).iloc[:, 1:].values.astype(float)
    dates = pd.to_datetime(pd.read_excel(os.path.join(ATT, '附件2.xlsx'), '小区负载').iloc[:, 0]).dt.normalize()
    return ld, pv, pr, dates

def audit(run_dir, settlement, load, pv, price, dates):
    print('=' * 70)
    print('AUDIT', os.path.basename(run_dir), settlement)
    errors = []
    for tag, n_win in [('Q42', 145), ('Q43', 148)]:
        rec = json.load(open(os.path.join(run_dir, tag + '.json'), encoding='utf-8'))
        n = len(rec)
        # 1) 独立费用重算
        for r in rec:
            p = price[np.where(dates == pd.Timestamp(r['date']))[0][0]]
            q0 = np.array(r['q0']); a = np.array(r['qfinal']); g = np.array(r['g'])
            up = np.maximum(a - q0, 0); dn = np.maximum(q0 - a, 0)
            plan = float(p @ q0)
            adj = float(p @ (1.5 * up + (-0.5 if settlement == 'refund' else 0.5) * dn))
            emg = float(5 * p @ g)
            for name, calc, stored in [('plan_fee', plan, r['plan_fee']), ('adj_fee', adj, r['adj_fee']),
                                       ('emg_fee', emg, r['emg_fee']), ('total', plan + adj + emg, r['total'])]:
                if abs(calc - stored) > 1e-6:
                    errors.append('%s %s %s: calc %.6f vs stored %.6f' % (tag, r['date'], name, calc, stored))
        # 2) SOC 跨日连续
        for i in range(1, n):
            if abs(rec[i]['E_in'] - rec[i - 1]['E_end']) > 1e-9:
                errors.append('%s %s SOC chain break: %.9f' % (tag, rec[i]['date'], rec[i]['E_in'] - rec[i - 1]['E_end']))
        # 3) 物理边界与互斥（每个轨迹）
        max_overlap, max_bal_viol, min_E, max_E = 0, 0, 1e9, -1e9
        for r in rec:
            c = np.array(r['c']); rr = np.array(r['r']); gg = np.array(r['g']); E = np.array(r['E'])
            a = np.array(r['qfinal'])
            day_idx = np.where(dates == pd.Timestamp(r['date']))[0][0]
            L = load[day_idx] * (1 / 6); PV = pv[day_idx] * (1 / 6)
            max_overlap = max(max_overlap, float(np.max(np.minimum(c, rr))))
            viol = PV + a + rr + gg - L - c
            max_bal_viol = max(max_bal_viol, float(np.min(viol)))
            min_E = min(min_E, float(E.min())); max_E = max(max_E, float(E.max()))
            if float(c.max()) > 833.34 or float(rr.max()) > 833.34 or float(gg.min()) < -1e-12:
                errors.append('%s %s power/emergency bound violation' % (tag, r['date']))
        print('  %s: days=%d, min_E=%.4f max_E=%.4f max_overlap=%.3g max_balance_viol=%.3g'
              % (tag, n, min_E, max_E, max_overlap, max_bal_viol))
        # 4) 评估窗口
        ev = [r for r in rec if '2025-02-01' <= r['date'] <= '2025-12-31']
        print('  %s: evaluation days=%d (expect 334)' % (tag, len(ev)))
        if len(ev) != 334:
            errors.append('%s evaluation window %d != 334' % (tag, len(ev)))
        tot = {k: sum(r[k] for r in ev) for k in ['plan_fee', 'adj_fee', 'emg_fee', 'total']}
        emg_kwh = sum(np.array(r['g']).sum() for r in ev)
        esc = sum(r.get('milp_escalations', 0) for r in ev)
        print('  %s: 总费用=%.2f (计划%.2f 调整%.2f 紧急%.2f) 紧急电量=%.2f kWh MILP升级=%d'
              % (tag, tot['total'], tot['plan_fee'], tot['adj_fee'], tot['emg_fee'], emg_kwh, esc))
        if not errors:
            print('  %s: 全部通过' % tag)
    return errors

ld, pv, price, dates = load_actual()
all_errors = []
all_errors += audit(os.path.join(BASE, '结果_refund'), 'refund', ld, pv, price, dates)
all_errors += audit(os.path.join(BASE, '结果_strict'), 'strict', ld, pv, price, dates)

# 跨口径与跨问题方向性
def totals(d, tag):
    rec = json.load(open(os.path.join(BASE, d, tag + '.json'), encoding='utf-8'))
    ev = [r for r in rec if '2025-02-01' <= r['date'] <= '2025-12-31']
    return sum(r['total'] for r in ev), sum(np.array(r['g']).sum() for r in ev)

for d in ['结果_refund', '结果_strict']:
    t42, e42 = totals(d, 'Q42'); t43, e43 = totals(d, 'Q43')
    print('%-14s Q42=%.2f (紧急%.0f)  Q43=%.2f (紧急%.0f)  Q43-Q42=%.2f'
          % (d, t42, e42, t43, e43, t43 - t42))
    if t43 > t42:
        all_errors.append('%s Q43 total > Q42 total (方向异常)' % d)
print()
print('旧口径参考: 4-2 约16,175,855元 4-3 约15,981,318元 (仅作量级对比,不作为结果)')
print('总错误数:', len(all_errors))
for e in all_errors[:30]:
    print('  ERR:', e)
print('AUDIT', 'PASS' if not all_errors else 'FAIL')
