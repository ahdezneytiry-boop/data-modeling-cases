# -*- coding: utf-8 -*-
"""
第四问统一时间窗口 MILP/LP 内核 (HiGHS)
单位 kWh。三种窗口模式:
  'plan'   : 自由决策购电量 q
  'adjust' : 相对 q_base 的上下调 up/dn, 最终购电 a=qbase+up-dn
  'exec'   : 购电量锁定 q_fix, 仅调度储能
窗口 [t0,t1); 进入电量 E_in; 终端残值 lam 作用于窗口末 E。
use_mutex=True 时引入 0-1 变量 y_t 强制 c_t·r_t=0 (充放电互斥):
  c_t<=PMAX*y_t ; r_t<=PMAX*(1-y_t)
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from data_core import T, ETA_C, ETA_D, PMAX, EMAX, EMIN


def solve_window(mode, p, Lkwh, PVkwh, E_in, t0=0, t1=T,
                 q_base=None, q_fix=None, lam=0.0,
                 k_up=1.5, k_dn=0.5, E_end=None, use_mutex=True,
                 settlement='refund'):
    if settlement not in ('refund', 'strict'):
        raise ValueError('settlement must be refund or strict')
    idx = np.arange(t0, t1)
    nl = len(idx)
    pb = p[idx]; Lb = Lkwh[idx]; PVb = PVkwh[idx]

    if mode == 'plan':
        blocks = ['q','c','r','g','E']; off = {b:i for i,b in enumerate(blocks)}
        ncore=5
        def V(b,i): return off[b]*nl+i
        cvec = np.zeros(ncore*nl)
        cvec[[V('q',i) for i in range(nl)]] = pb
        cvec[[V('g',i) for i in range(nl)]] = 5*pb
        if E_end is None: cvec[V('E',nl-1)] -= lam
        bounds=[(0,None)]*(ncore*nl)
        for i in range(nl):
            bounds[V('c',i)]=(0,PMAX); bounds[V('r',i)]=(0,PMAX); bounds[V('E',i)]=(EMIN,EMAX)
    elif mode == 'adjust':
        blocks=['up','dn','c','r','g','E']; off={b:i for i,b in enumerate(blocks)}
        ncore=6
        def V(b,i): return off[b]*nl+i
        qb=q_base[idx]
        cvec=np.zeros(ncore*nl)
        cvec[[V('up',i) for i in range(nl)]]=k_up*pb
        cvec[[V('dn',i) for i in range(nl)]]=(-1 if settlement=='refund' else 1)*k_dn*pb
        cvec[[V('g',i) for i in range(nl)]]=5*pb
        if E_end is None: cvec[V('E',nl-1)] -= lam
        bounds=[(0,None)]*(ncore*nl)
        for i in range(nl):
            bounds[V('c',i)]=(0,PMAX); bounds[V('r',i)]=(0,PMAX); bounds[V('E',i)]=(EMIN,EMAX)
            bounds[V('dn',i)]=(0,qb[i])
    elif mode == 'exec':
        blocks=['c','r','g','E']; off={b:i for i,b in enumerate(blocks)}
        ncore=4
        def V(b,i): return off[b]*nl+i
        cvec=np.zeros(ncore*nl)
        cvec[[V('g',i) for i in range(nl)]]=5*pb
        if E_end is None: cvec[V('E',nl-1)] -= lam
        bounds=[(0,PMAX)]*nl+[(0,PMAX)]*nl+[(0,None)]*nl+[(EMIN,EMAX)]*nl
    else:
        raise ValueError(mode)

    nv0=ncore*nl
    # 追加二进制 y 块
    if use_mutex:
        nv=nv0+nl
        cvec=np.concatenate([cvec,np.zeros(nl)])
        bounds=bounds+[(0,1)]*nl
        def Y(i): return nv0+i
        integrality=np.zeros(nv); integrality[[Y(i) for i in range(nl)]]=1
    else:
        nv=nv0; integrality=None
        def Y(i): return None

    # ---- SOC 等式 ----
    rows,cols,vals,beq=[],[],[],[]
    for i in range(nl):
        rows+=[i,i,i]; cols+=[V('E',i),V('c',i),V('r',i)]; vals+=[1,-ETA_C,1/ETA_D]
        if i>0:
            rows+=[i]; cols+=[V('E',i-1)]; vals+=[-1]
        beq.append(E_in if i==0 else 0.0)
    Aeq=sp.csr_matrix((vals,(rows,cols)),shape=(nl,nv)); beq=np.array(beq)
    if E_end is not None:
        rE=np.zeros(nv); rE[V('E',nl-1)]=1.0
        Aeq=sp.vstack([Aeq,sp.csr_matrix(rE)]); beq=np.append(beq,E_end)

    # ---- 能量平衡不等式 ----
    ru,cu,vu=[],[],[]
    if mode=='plan':
        net=Lb-PVb
        for i in range(nl):
            ru+=[i]*4;cu+=[V('q',i),V('c',i),V('r',i),V('g',i)];vu+=[-1,1,-1,-1]
        nub=nl
    elif mode=='adjust':
        net=Lb-PVb-qb
        for i in range(nl):
            ru+=[i]*5;cu+=[V('up',i),V('dn',i),V('c',i),V('r',i),V('g',i)];vu+=[-1,1,1,-1,-1]
        nub=nl
    else:
        qb=q_fix[idx]; net=Lb-PVb-qb
        for i in range(nl):
            ru+=[i]*3;cu+=[V('c',i),V('r',i),V('g',i)];vu+=[1,-1,-1]
        nub=nl
    bub=-net

    # ---- 充放电互斥不等式 ----
    if use_mutex:
        base=nub
        rhs_mutex=np.zeros(2*nl)
        for i in range(nl):
            # c - PMAX*y <=0
            ru += [base+2*i]*2; cu+=[V('c',i),Y(i)]; vu+=[1,-PMAX]
            # r + PMAX*y <= PMAX
            ru += [base+2*i+1]*2; cu+=[V('r',i),Y(i)]; vu+=[1,PMAX]
            rhs_mutex[2*i+1]=PMAX
        bub=np.concatenate([bub,rhs_mutex])
    Aub=sp.csr_matrix((vu,(ru,cu)),shape=(len(bub),nv))

    kwargs=dict(method='highs',
                options={'mip_feasibility_tolerance':1e-8,
                         'primal_feasibility_tolerance':1e-9,
                         'dual_feasibility_tolerance':1e-9})
    if integrality is not None: kwargs['integrality']=integrality
    res=linprog(cvec,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=bounds,**kwargs)
    if not res.success:
        return dict(success=False,msg=res.message,status=res.status)
    x=res.x
    out=dict(success=True,idx=idx,cost=res.fun)
    for b in blocks:
        arr=x[[V(b,i) for i in range(nl)]]
        arr[np.abs(arr)<1e-7]=0   # 仅清除真零噪声, 保留MILP严格可行解
        out[b]=arr
    # E 直接采用求解器解(严格满足SOC等式), 不二次递推
    out['E']=out['E']
    if mode=='plan':
        out['a']=out['q']
    elif mode=='adjust':
        out['a']=qb+out['up']-out['dn']; out['a'][out['a']<1e-9]=0
    else:
        out['a']=q_fix[idx]
    return out
