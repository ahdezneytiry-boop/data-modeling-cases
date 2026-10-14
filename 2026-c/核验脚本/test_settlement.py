# -*- coding: utf-8 -*-
"""Settlement formula unit tests (from README hand examples + extra cases)."""
import sys, os
sys.path.insert(0, r'C:\Users\ahdezneytiry-boop1\.zcode\workspace\default\q4run\源代码')
import numpy as np
from run_causal import cash_cost

def t(price, q0, final, g, settlement, expect_total):
    p = np.array([price], dtype=float)
    r = cash_cost(p, np.array([q0]), np.array([final]), np.array([g]), settlement)
    ok = abs(r['total'] - expect_total) < 1e-9
    print('%s price=%g q0=%g final=%g g=%g -> total=%.10f expect=%g %s'
          % (settlement, price, q0, final, g, r['total'], expect_total, 'PASS' if ok else 'FAIL'))
    return ok

res = []
# README hand example: price=2, q0=10, final=6, no emergency -> refund 16, strict 24
res.append(t(2, 10, 6, 0, 'refund', 16.0))
res.append(t(2, 10, 6, 0, 'strict', 24.0))
# up-only: q0=10 -> final=12: refund=20+1.5*2*2=26; strict same
res.append(t(2, 10, 12, 0, 'refund', 26.0))
res.append(t(2, 10, 12, 0, 'strict', 26.0))
# emergency: g=3 -> emg=5*2*3=30
res.append(t(2, 10, 10, 3, 'refund', 50.0))
# up+down+emergency mixed: q0=10, final=11, g=2: refund=20+3*1+... up=1->1.5*2*1=3; total=20+3+5*2*2=43
res.append(t(2, 10, 11, 2, 'refund', 43.0))
res.append(t(2, 10, 11, 2, 'strict', 43.0))
# q0=10 final=8 g=2 refund: plan=20, dn=2 -> adj=-0.5*2*2=-2, emg=20 -> 38; strict: adj=+2 -> 42
res.append(t(2, 10, 8, 2, 'refund', 38.0))
res.append(t(2, 10, 8, 2, 'strict', 42.0))
print('ALL PASS' if all(res) else 'SOME FAILED')
