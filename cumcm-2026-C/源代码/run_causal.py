"""Minimal causal wrapper around the original HiGHS MILP.

Forecasts are built from completed days and previous observations only.
The simulation reveals current actuals immediately before first-step control.
No rolling/cyclic column permutation. Old result files are never read.
"""
from pathlib import Path
from datetime import datetime
import argparse
import json
import numpy as np
import data_core as dc
from optimizer import solve_window


def checked(result):
    if not result['success']:
        raise RuntimeError(result)
    return result


# LP-first with MILP escalation (matches the documented Q2/Q3 solver strategy):
# solve the LP relaxation without mutex binaries; if any segment has
# simultaneous charge and discharge, re-solve the window with 0-1 variables.
# An LP optimum that satisfies mutex is optimal for the MILP (adding integrality
# constraints cannot improve it), so results are mathematically identical.
_MILP_COUNT = {'n': 0}

def solve_opt(mode, p, Lkwh, PVkwh, E_in, **kw):
    res = solve_window(mode, p, Lkwh, PVkwh, E_in, use_mutex=False, **kw)
    if not res['success']:
        return res
    c = res.get('c')
    r = res.get('r')
    if c is not None and r is not None and float(np.max(np.minimum(c, r))) > 1e-9:
        _MILP_COUNT['n'] += 1
        res = solve_window(mode, p, Lkwh, PVkwh, E_in, use_mutex=True, **kw)
    return res



def day_baselines(load_history, price_history, dates_history, target_date,
                  initial_load, initial_price):
    if len(load_history) == 0:
        return np.full(dc.T, initial_load), np.full(dc.T, initial_price)
    weekday = target_date.weekday()
    chosen = [i for i, day in enumerate(dates_history) if day.weekday() == weekday]
    lp = np.mean(load_history[chosen[-4:]] if chosen else load_history[-7:], axis=0)
    pp = price_history[0].copy()
    for daily in price_history[1:]:
        pp = 0.5 * daily + 0.5 * pp
    return lp, pp


def platform_forecast(records, day, step):
    now = datetime.fromisoformat(str(day.date())) .replace(hour=step//6, minute=step%6*10)
    eligible = []
    for record in records:
        if record['target_date'] != str(day.date()):
            continue
        issued = datetime.fromisoformat(record['issued_at'])
        trained = datetime.fromisoformat(record['training_end'])
        if issued.tzinfo or trained.tzinfo:
            raise ValueError('Use consistent naive local timestamps for all records')
        if trained >= issued:
            raise ValueError('training_end must precede issued_at')
        if issued <= now:
            values = np.asarray(record['values'], float)
            if values.shape != (dc.T,) or not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError('Forecast needs 144 finite positive prices in interval order')
            source = record['source']
            eligible.append((issued, values, source))
    if not eligible:
        return None
    return max(eligible, key=lambda item:item[0])


def forecasts(base_load, base_price, past_load, past_price, pv_forecast):
    """Inputs contain past observations only, never future actual arrays."""
    t = len(past_load)
    left = max(0, t-12)
    lp, pp = base_load.copy(), base_price.copy()
    if t:
        lp += np.mean(past_load[left:t] - base_load[left:t])
        pp += np.mean(past_price[left:t] - base_price[left:t])
    return np.maximum(lp,0)*dc.DT, np.maximum(pp,1e-6), np.maximum(pv_forecast,0)*dc.DT


def cash_cost(price, q0, final, emergency, settlement):
    up = np.maximum(final-q0,0)
    down = np.maximum(q0-final,0)
    plan = float(price @ q0)
    adj = float(price @ (1.5*up + (-0.5 if settlement=='refund' else 0.5)*down))
    emg = float(5*price @ emergency)
    return dict(plan_fee=plan, adj_fee=adj, emg_fee=emg, total=plan+adj+emg)


def run_day(actual_load, actual_price, actual_pv, pv_by_release,
            base_load, base_price, initial_energy, adjust, settlement='refund',
            alpha=0.5, platform_records=(), day=None):
    """The outer simulator holds actuals; only slices through t reach controller."""
    selected = platform_forecast(platform_records,day,0) if platform_records else None
    plan_price = selected[1] if selected else base_price
    Lhat, phat, pvhat = forecasts(base_load,plan_price,np.array([]),np.array([]),pv_by_release[0])
    lam = alpha * float(np.median(phat))
    pl = checked(solve_opt('plan',phat,Lhat,pvhat,initial_energy,lam=lam,settlement=settlement))
    q0 = pl['a'].copy(); final=q0.copy(); state=float(initial_energy)
    result = {k:np.zeros(dc.T) for k in ['c','r','g','E']}
    source_log = []
    m0 = _MILP_COUNT['n']
    for t in range(dc.T):
        release=(t//36)*6
        selected = platform_forecast(platform_records,day,t) if platform_records else None
        pred_price=selected[1] if selected else base_price
        Lhat, phat, pvhat=forecasts(base_load,pred_price,
            actual_load[:t].copy(),actual_price[:t].copy(),pv_by_release[release])
        source_log.append(str(selected[2]) if selected else 'SES_history')
        if adjust and t in (36,72,108):
            new=checked(solve_opt('adjust',phat,Lhat,pvhat,state,t0=t,t1=dc.T,
                q_base=q0,lam=lam,settlement=settlement))
            final[t:]=new['a']
        # Exactly the current interval is observed before control.
        Lhat[t]=actual_load[t]*dc.DT
        phat[t]=actual_price[t]
        pvhat[t]=actual_pv[t]*dc.DT
        ex=checked(solve_opt('exec',phat,Lhat,pvhat,state,t0=t,t1=dc.T,
            q_fix=final,lam=lam,settlement=settlement))
        for key in result: result[key][t]=ex[key][0]
        state=state+dc.ETA_C*result['c'][t]-result['r'][t]/dc.ETA_D
        if abs(state-result['E'][t])>1e-5:
            raise AssertionError('SOC recursion mismatch')
    result.update(q0=q0,qfinal=final,E_in=initial_energy,E_end=state,lam=lam,
        alpha=alpha,forecast_sources=source_log,
        milp_escalations=_MILP_COUNT['n']-m0,
        up=np.maximum(final-q0,0),dn=np.maximum(q0-final,0))
    result.update(cash_cost(actual_price,q0,final,result['g'],settlement))
    validate(result,actual_load,actual_pv)
    return result


def validate(v,load,pv):
    tol=1e-5
    c,r,g,E,q=(v[k] for k in ['c','r','g','E','qfinal'])
    assert all(np.isfinite(x).all() for x in [c,r,g,E,q])
    assert all((x>=-tol).all() for x in [c,r,g,q])
    assert (c<=dc.PMAX+tol).all() and (r<=dc.PMAX+tol).all()
    assert ((E>=dc.EMIN-tol)&(E<=dc.EMAX+tol)).all()
    assert not ((c>tol)&(r>tol)).any()
    assert (pv*dc.DT+q+r+g-load*dc.DT-c>=-tol).all()
    recomputed=v['E_in']+np.cumsum(dc.ETA_C*c-r/dc.ETA_D)
    assert np.max(np.abs(recomputed-E))<tol


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attachments',required=True)
    parser.add_argument('--output',required=True,help='New output directory; existing results are not overwritten')
    parser.add_argument('--initial-load-kw',type=float,required=True,help='Explicit pre-sample assumption, not estimated from future data')
    parser.add_argument('--initial-price',type=float,required=True)
    parser.add_argument('--days',type=int,default=365)
    parser.add_argument('--settlement',choices=['refund','strict'],default='refund')
    parser.add_argument('--alpha',type=float,default=0.5)
    parser.add_argument('--platform-json',help='JSON list with issued_at, training_end, target_date, source, values[144]')
    args=parser.parse_args()
    if args.initial_load_kw<0 or args.initial_price<=0 or not 0<=args.alpha<=1 or args.days<1:
        parser.error('Invalid initialization, alpha, or day count')
    dc.ATT=str(Path(args.attachments).resolve())
    load,pv,dates=dc.load_attachment2();price,pdates=dc.load_attachment4()
    if not dates.equals(pdates):raise ValueError('Attachment date mismatch')
    fc=dc.load_attachment3()
    if not np.isfinite(load).all() or not np.isfinite(pv).all() or not np.isfinite(price).all() or (price<=0).any():
        raise ValueError('Invalid actual data')
    platform=json.loads(Path(args.platform_json).read_text(encoding='utf-8')) if args.platform_json else []
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    totals={}
    for label,adjust in [('Q42',False),('Q43',True)]:
        records=[];energy=dc.E0_INIT
        for d in range(min(args.days,len(dates))):
            day=dates.iloc[d]
            lb,pb=day_baselines(load[:d],price[:d],dates.iloc[:d],day,args.initial_load_kw,args.initial_price)
            fday={h:dc.get_forecast_series(fc,day,h) for h in [0,6,12,18]}
            v=run_day(load[d],price[d],pv[d],fday,lb,pb,energy,adjust,args.settlement,args.alpha,platform,day)
            energy=v['E_end'];v['date']=str(day.date());records.append(v)
            print(label,v['date'],round(v['total'],2),flush=True)
        (out/f'{label}.json').write_text(json.dumps(records,ensure_ascii=False,default=serial),encoding='utf-8')
        evaluation=[v for v in records if '2025-02-01'<=v['date']<='2025-12-31']
        totals[label]={k:sum(v[k] for v in evaluation) for k in ['plan_fee','adj_fee','emg_fee','total']}
        totals[label]['evaluation_days']=len(evaluation)
        totals[label]['E_end']=energy
    totals['configuration']=vars(args)
    totals['timeline']='144 right-endpoint columns in source order; no rotation'
    totals['status']='complete_334_days' if all(totals[x]['evaluation_days']==334 for x in ['Q42','Q43']) else 'partial_run_not_annual_result'
    (out/'summary.json').write_text(json.dumps(totals,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
