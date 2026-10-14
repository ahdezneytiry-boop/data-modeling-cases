# -*- coding: utf-8 -*-
import json, os
base = r'C:\Users\ahdezneytiry-boop1\.zcode\workspace\default\q4run'
for s in ['refund', 'strict']:
    helper = os.path.join(base, '结果_' + s + '_q43', 'Q43.json')
    main = os.path.join(base, '结果_' + s, 'Q43.json')
    print(s, 'helper exists:', os.path.exists(helper), 'main exists:', os.path.exists(main))
    if os.path.exists(helper) and os.path.exists(main):
        h = json.load(open(helper, encoding='utf-8'))
        m = json.load(open(main, encoding='utf-8'))
        same = json.dumps(h, sort_keys=True) == json.dumps(m, sort_keys=True)
        print(s, 'helper days:', len(h), 'main days:', len(m), 'BIT-IDENTICAL:', same)
        if not same:
            for i, (a, b) in enumerate(zip(h, m)):
                if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
                    print('  first diff at day', i, a.get('date'), '|', b.get('date'))
                    break
print('CROSS-CHECK DONE')
