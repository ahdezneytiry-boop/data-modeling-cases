# -*- coding: utf-8 -*-
"""
第四问核心数据模块：统一加载附件2/3/4，对齐到自然时间顺序（0:00-0:10 为第0时段）
- 原始列 [00:10,...,23:50,0:00+1] 为时段右端点，保持原顺序。
- 第0列对应00:00-00:10，末列对应23:50-24:00，不循环旋转。
- 附件3整点24h预报 -> 线性插值到144个10分钟时段
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os

ATT = os.environ.get('Q4_ATTACHMENTS', str(Path(__file__).resolve().parents[1] / '附件'))
T = 144                       # 每天144个10分钟时段
DT = 10.0 / 60.0             # 每个时段时长(h)
ETA_C = 0.9                  # 充电效率
ETA_D = 0.9                  # 放电效率
PMAX = 5000.0 * DT           # 单时段最大充/放电量 kWh = 833.333
EMAX, EMIN = 10800.0, 1200.0 # SOC上下限
ECAP = 12000.0
E0_INIT = 6000.0             # 2025-01-01 0:00 初始电量


def _reorder_natural(arr145):
    """144个时段右端点列，原序即自然时序。"""
    a = np.asarray(arr145, dtype=float)
    if a.shape != (T,) or not np.isfinite(a).all():
        raise ValueError('Expected 144 finite interval values')
    return a.copy()


def load_attachment2():
    """返回 load(365,144)kW, pv_real(365,144)kW, 日期列表"""
    xl = pd.ExcelFile(f"{ATT}/附件2.xlsx")
    ld = pd.read_excel(xl, "小区负载")
    pv = pd.read_excel(xl, "光伏发电实际功率")
    dates = pd.to_datetime(ld.iloc[:, 0]).dt.normalize()
    load = np.array([_reorder_natural(row) for row in ld.iloc[:, 1:].values])
    pvr = np.array([_reorder_natural(row) for row in pv.iloc[:, 1:].values])
    return load, pvr, dates


def load_attachment4():
    """返回 price(365,144) 元/kWh（波动电价）"""
    pr = pd.read_excel(f"{ATT}/附件4.xlsx")
    dates = pd.to_datetime(pr.iloc[:, 0]).dt.normalize()
    price = np.array([_reorder_natural(row) for row in pr.iloc[:, 1:].values])
    return price, dates


def load_attachment3():
    """解析附件3，返回 forecast[day][H] = 长度144的预测功率(kW)
    H in {0,6,12,18} 为发布时刻。预报k小时对应绝对时刻 H+k 点。
    用整点值在10分钟网格线性插值。"""
    raw = pd.read_excel(f"{ATT}/附件3.xlsx")
    # 向下填充日期
    raw['日期'] = raw['日期'].ffill()
    raw['日期'] = pd.to_datetime(raw['日期'])
    fcols = [c for c in raw.columns if str(c).startswith('预报') and '小时' in str(c)]  # 24列
    Hmap = {'0:00': 0, '6:00': 6, '12:00': 12, '18:00': 18}
    forecast = {}
    for _, r in raw.iterrows():
        d = r['日期'].normalize()
        H = Hmap[str(r['预报时刻']).strip()]
        hourly = np.array(r[fcols].values, dtype=float)  # 预报1h..24h -> 绝对 H+1..H+24 点
        # 用 (绝对钟点 -> 功率) 字典; 仅保留当天0..24需要的点, 但插值允许跨夜
        pts = {}
        for k in range(1, 25):
            pts[H + k] = hourly[k - 1]
        # 发布时刻H本身: 用相邻预报点线性外插一个锚点(夜间为0附近)
        xs_all = np.array(sorted(pts.keys()))
        ys_all = np.array([pts[x] for x in xs_all])
        # 对全天整点 0..24 插值(边界用最近值外插)
        xs = np.arange(25)
        hour_full = np.interp(xs, xs_all, ys_all)
        # 映射到10分钟网格: t=0..143, 钟点 frac=t/6
        tt = (np.arange(144) + 1) / 6.0
        h0 = np.floor(tt).astype(int)
        grid = hour_full[h0] + (tt - h0) * (hour_full[np.minimum(h0 + 1, 24)] - hour_full[h0])
        forecast[(d, H)] = grid
    return forecast


def get_forecast_series(forecast, day, H, n=144):
    """取某day、发布时刻H对当天0:00起n个10分钟点的预测功率(kW)。
    若预报覆盖到次日(如18:00预报)，当天部分直接取0..143。"""
    return forecast[(day, H)][:n]


if __name__ == "__main__":
    load, pvr, dates = load_attachment2()
    price, _ = load_attachment4()
    fc = load_attachment3()
    print("load", load.shape, "pv", pvr.shape, "price", price.shape, "forecast blocks", len(fc))
    print("日期范围:", dates.iloc[0], "->", dates.iloc[-1], "天数", len(dates))
    # 校验自然顺序: 1月1日 0:00-0:10 光伏应为0, 正午12点附近最大
    print("\n附件2 1月1日光伏 自然顺序 t=0(0:00):", pvr[0,0], " t=72(12:00):", pvr[0,72], " t=78(13:00):", pvr[0,78])
    print("附件4 1月1日电价 t=0..3:", price[0,:4])
    # 校验预报
    f0 = get_forecast_series(fc, dates.iloc[0], 0)
    print("\n0:00预报 1月1日 t=0:", f0[0], "t=72(12点):", f0[72], "t=78(13点):", f0[78])
    f6 = get_forecast_series(fc, dates.iloc[0], 6)
    print("6:00预报 1月1日 t=36(6点):", f6[36], "t=72(12点):", f6[72])
    # 能量量级
    print("\n负载功率范围:", load.min(), load.max())
    print("光伏功率范围:", pvr.min(), pvr.max())
    print("电价范围:", price.min(), price.max(), "均值:", price.mean())
    # 单时段最大充放
    print("PMAX(kWh/10min):", PMAX)
