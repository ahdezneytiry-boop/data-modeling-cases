# -*- coding: utf-8 -*-
"""Determinism check: run the same 3-day slice twice, compare bit-for-bit."""
import subprocess, json, os, sys

code = r'C:\Users\ahdezneytiry-boop1\.zcode\workspace\default\q4run\源代码\run_causal.py'
att = r'C:\Users\ahdezneytiry-boop1\.zcode\workspace\default\q4run\附件'
outs = [r'C:\Users\ahdezneytiry-boop1\.zcode\workspace\default\q4run\det_a',
        r'C:\Users\ahdezneytiry-boop1\.zcode\workspace\default\q4run\det_b']
for o in outs:
    if os.path.exists(o):
        import shutil; shutil.rmtree(o)
for o in outs:
    r = subprocess.run([sys.executable, code, '--attachments', att, '--output', o,
                        '--initial-load-kw', '4626.03', '--initial-price', '0.7662',
                        '--days', '3', '--settlement', 'refund'],
                       capture_output=True, text=True)
    print(o, 'rc=', r.returncode, r.stderr[-200:] if r.returncode else 'ok')

a = json.load(open(os.path.join(outs[0], 'Q42.json'), encoding='utf-8'))
b = json.load(open(os.path.join(outs[1], 'Q42.json'), encoding='utf-8'))
same = json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
print('Q42 deterministic:', same)
a3 = json.load(open(os.path.join(outs[0], 'Q43.json'), encoding='utf-8'))
b3 = json.load(open(os.path.join(outs[1], 'Q43.json'), encoding='utf-8'))
print('Q43 deterministic:', json.dumps(a3, sort_keys=True) == json.dumps(b3, sort_keys=True))
