# -*- coding: utf-8 -*-
"""
Тема 2. Планирование раскроя материала (одномерная задача, 1D-CSP).

Содержит:
  1. нижние границы (длинная и LP-релаксация Гилмора-Гомори);
  2. жадный алгоритм First Fit Decreasing (FFD);
  3. генерацию всех допустимых схем раскроя одного листа;
  4. полный перебор покрытия спроса схемами (рекурсия с запоминанием);
  5. точное решение целочисленной задачи (HiGHS) как эталон;
  6. анализ чувствительности и графики.
"""
import itertools
import json
import math
import sys
from functools import lru_cache

import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Исходные данные
# ----------------------------------------------------------------------------
L0 = 100
LENGTHS = [45, 35, 28, 20, 12]
DEMAND = [12, 15, 18, 22, 30]


# ----------------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------------
def total_length(lengths, demand):
    """Суммарная полезная длина всех требуемых деталей."""
    return sum(l * b for l, b in zip(lengths, demand))


def waste_stats(n_sheets, L, lengths, demand):
    """Суммарные отходы и их доля (в %) для n_sheets листов длины L."""
    total = n_sheets * L
    waste = total - total_length(lengths, demand)
    return waste, 100.0 * waste / total


# ----------------------------------------------------------------------------
# 1. Нижние границы
# ----------------------------------------------------------------------------
def lower_bound_simple(L, lengths, demand, w=0):
    """Нижняя граница по суммарной длине: ceil(sum(l_i*b_i) / L).
    С пропилом w деталь «занимает» l_i + w, а лист – L + w."""
    s = sum((l + w) * b for l, b in zip(lengths, demand))
    return s / (L + w), math.ceil(s / (L + w) - 1e-9)


def generate_patterns(L, lengths, w=0, maximal_only=False):
    """Все допустимые схемы раскроя одного листа.
    Схема – вектор (n_1,...,n_m), где n_i – число деталей типа i на листе,
    и  sum n_i*(l_i + w) <= L + w.  Пустая схема исключена."""
    m = len(lengths)
    eff = [l + w for l in lengths]
    cap = L + w
    patterns = []

    def rec(i, remaining, cur):
        if i == m:
            if any(cur):
                patterns.append(tuple(cur))
            return
        for k in range(remaining // eff[i] + 1):
            cur.append(k)
            rec(i + 1, remaining - k * eff[i], cur)
            cur.pop()

    rec(0, cap, [])
    if maximal_only:
        # схема максимальна, если в неё нельзя добавить ни одной детали
        def used(p):
            return sum(n * e for n, e in zip(p, eff))
        patterns = [p for p in patterns
                    if all(used(p) + e > cap for e in eff)]
    return patterns


def lp_relaxation(L, lengths, demand, w=0):
    """Непрерывная релаксация задачи Гилмора-Гомори:
         min sum x_p,  sum_p n_ip x_p >= b_i,  x_p >= 0.
    Возвращает (оптимум LP, число схем, решение x, двойственные оценки)."""
    pats = generate_patterns(L, lengths, w, maximal_only=True)
    A = np.array(pats, dtype=float).T          # m x P
    c = np.ones(A.shape[1])
    res = linprog(c, A_ub=-A, b_ub=-np.array(demand, dtype=float),
                  bounds=(0, None), method="highs")
    duals = -res.ineqlin.marginals             # цены деталей y_i
    return res.fun, len(pats), pats, res.x, duals


# ----------------------------------------------------------------------------
# 2. First Fit Decreasing
# ----------------------------------------------------------------------------
def ffd(L, lengths, demand, w=0):
    """Первый подходящий в порядке убывания.
    Возвращает список листов; лист – список длин деталей (в порядке укладки)."""
    items = []
    for l, b in zip(lengths, demand):
        items += [l] * b
    items.sort(reverse=True)
    sheets, free = [], []            # free[j] – свободная длина листа j
    for it in items:
        need = it + w
        for j in range(len(sheets)):
            if free[j] >= need - 1e-12:
                sheets[j].append(it)
                free[j] -= need
                break
        else:
            sheets.append([it])
            free.append(L + w - need)
    return sheets


def check_sheets(sheets, L, lengths, demand, w=0):
    """Проверка допустимости: каждый лист помещается, спрос выполнен точно."""
    for s in sheets:
        assert sum(s) + w * (len(s) - 1) <= L + 1e-9, "лист переполнен"
    for l, b in zip(lengths, demand):
        assert sum(s.count(l) for s in sheets) == b, "спрос не выполнен"
    return True


# ----------------------------------------------------------------------------
# 3. Полный перебор на подмножестве типов деталей
# ----------------------------------------------------------------------------
def brute_force_cover(L, lengths, demand, w=0):
    """Точный перебор: минимум листов, покрывающих спрос,
    как комбинация схем раскроя. Рекурсия по вектору остаточного спроса
    с запоминанием состояний (memoization)."""
    pats = generate_patterns(L, lengths, w, maximal_only=True)
    # чем «плотнее» схема, тем раньше её пробуем (для скорости)
    eff = [l + w for l in lengths]
    pats.sort(key=lambda p: -sum(n * e for n, e in zip(p, eff)))

    @lru_cache(maxsize=None)
    def best(rem):
        if not any(rem):
            return 0, ()
        top = None
        for p in pats:
            new = tuple(max(0, r - n) for r, n in zip(rem, p))
            if new == rem:
                continue
            cnt, path = best(new)
            cand = (cnt + 1, (p,) + path)
            if top is None or cand[0] < top[0]:
                top = cand
        return top

    return best(tuple(demand)), pats


def exact_ilp(L, lengths, demand, w=0):
    """Точное целочисленное решение (HiGHS) на множестве максимальных схем."""
    pats = generate_patterns(L, lengths, w, maximal_only=True)
    A = np.array(pats, dtype=float).T
    P = A.shape[1]
    res = milp(c=np.ones(P),
               constraints=LinearConstraint(A, lb=np.array(demand, float),
                                            ub=np.inf),
               integrality=np.ones(P), bounds=Bounds(0, np.inf))
    x = np.round(res.x).astype(int)
    return int(round(res.fun)), pats, x


# ----------------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------------
COLORS = {45: "#4C78A8", 35: "#F58518", 28: "#54A24B",
          20: "#B279A2", 12: "#E45756"}


def draw_layout(ax, sheets, L, title, w=0, labels=None):
    """Горизонтальные отрезки: детали по порядку укладки + отход (серый)."""
    labels = labels or [f"Лист {i + 1}" for i in range(len(sheets))]
    for row, sheet in enumerate(sheets):
        y = len(sheets) - row - 1
        x = 0
        for it in sheet:
            ax.barh(y, it, left=x, height=0.6, color=COLORS.get(it, "#999"),
                    edgecolor="black", linewidth=0.8)
            ax.text(x + it / 2, y, str(it), ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
            x += it + w
        rest = L - x + w
        if rest > 1e-9:
            ax.barh(y, rest, left=x, height=0.6, color="#DDDDDD",
                    edgecolor="black", hatch="//", linewidth=0.8)
            lab = f"отход {rest:g}" if rest >= 12 else f"{rest:g}"
            ax.text(x + rest / 2, y, lab, ha="center", va="center",
                    fontsize=8 if rest >= 12 else 7)
    ax.set_xlim(0, L + w)
    ax.set_yticks(range(len(sheets)))
    ax.set_yticklabels([lab for lab in labels][::-1])
    ax.set_xlabel("Длина, ед.")
    ax.set_title(title)


def run_all(save_dir="."):
    out = {}
    L, lengths, demand = L0, LENGTHS, DEMAND

    # ---- 1. нижние границы -------------------------------------------------
    lb_real, lb_int = lower_bound_simple(L, lengths, demand)
    lp_val, n_pat, lp_pats, lp_x, duals = lp_relaxation(L, lengths, demand)
    out["total_len"] = total_length(lengths, demand)
    out["lb_real"], out["lb_int"] = lb_real, lb_int
    out["lp_val"], out["lp_int"] = lp_val, math.ceil(lp_val - 1e-9)
    out["n_maximal_patterns"] = n_pat
    out["n_all_patterns"] = len(generate_patterns(L, lengths))
    out["duals"] = [float(d) for d in duals]

    # ---- 2. FFD ------------------------------------------------------------
    sh = ffd(L, lengths, demand)
    check_sheets(sh, L, lengths, demand)
    out["ffd_sheets"] = len(sh)
    out["ffd_waste"], out["ffd_waste_pct"] = waste_stats(len(sh), L, lengths, demand)
    out["ffd_layout"] = [list(map(int, x)) for x in sh]

    # ---- 3. точный оптимум (ЦЛП) ------------------------------------------
    opt, pats, x = exact_ilp(L, lengths, demand)
    out["opt_sheets"] = opt
    out["opt_waste"], out["opt_waste_pct"] = waste_stats(opt, L, lengths, demand)
    out["opt_plan"] = [(list(map(int, pats[j])), int(x[j]))
                       for j in range(len(pats)) if x[j] > 0]

    # ---- 4. полный перебор на подмножестве (первые 3 типа) ----------------
    sub_l, sub_b = lengths[:3], demand[:3]
    (bf_cnt, bf_path), sub_pats = brute_force_cover(L, sub_l, sub_b)
    sub_ffd = ffd(L, sub_l, sub_b)
    sub_lb_real, sub_lb = lower_bound_simple(L, sub_l, sub_b)
    sub_lp = lp_relaxation(L, sub_l, sub_b)[0]
    out["sub"] = dict(
        total_len=total_length(sub_l, sub_b),
        n_all_patterns=len(generate_patterns(L, sub_l)),
        n_maximal_patterns=len(sub_pats),
        lb_real=sub_lb_real, lb_int=sub_lb, lp=sub_lp,
        ffd_sheets=len(sub_ffd),
        ffd_waste=waste_stats(len(sub_ffd), L, sub_l, sub_b)[0],
        ffd_waste_pct=waste_stats(len(sub_ffd), L, sub_l, sub_b)[1],
        bf_sheets=bf_cnt,
        bf_waste=waste_stats(bf_cnt, L, sub_l, sub_b)[0],
        bf_waste_pct=waste_stats(bf_cnt, L, sub_l, sub_b)[1],
        lb_waste=waste_stats(sub_lb, L, sub_l, sub_b)[0],
        lb_waste_pct=waste_stats(sub_lb, L, sub_l, sub_b)[1],
        bf_plan=[list(map(int, p)) for p in bf_path],
        top_patterns=[(list(p), L - sum(n * l for n, l in zip(p, sub_l)))
                      for p in sub_pats[:8]],
    )
    # «хорошие» схемы полного набора (отход <= 2)
    good = sorted(((L - sum(n * l for n, l in zip(p, lengths)), p)
                   for p in lp_pats))
    out["good_patterns"] = [(list(map(int, p)), int(r)) for r, p in good[:10]]

    # ---- 5. чувствительность ----------------------------------------------
    sens = {}

    def case(L_, lengths_, demand_, w_=0):
        lbr, lbi = lower_bound_simple(L_, lengths_, demand_, w_)
        lpv = lp_relaxation(L_, lengths_, demand_, w_)[0]
        f = len(ffd(L_, lengths_, demand_, w_))
        o = exact_ilp(L_, lengths_, demand_, w_)[0]
        tl = total_length(lengths_, demand_)
        res = dict(lb=lbi, lb_real=lbr, lp=lpv, ffd=f, opt=o, total=tl)
        for key, n in (("lb", lbi), ("ffd", f), ("opt", o)):
            wst = n * L_ - tl
            res[key + "_waste"] = wst
            res[key + "_pct"] = 100.0 * wst / (n * L_)
        return res

    sens["base"] = case(L, lengths, demand)
    sens["L90"] = case(90, lengths, demand)
    sens["L110"] = case(110, lengths, demand)
    d20 = list(demand)
    d20[0] = math.ceil(demand[0] * 1.2)
    d20[4] = math.ceil(demand[4] * 1.2)
    sens["demand20"] = case(L, lengths, d20)
    sens["demand20"]["demand"] = d20
    sens["kerf1"] = case(L, lengths, demand, 1)
    # по каждому типу отдельно: +20 % спроса
    per_type = []
    for i in range(5):
        d = list(demand)
        d[i] = math.ceil(d[i] * 1.2)
        c = case(L, lengths, d)
        per_type.append(dict(type=i + 1, demand=d[i], **c))
    sens["per_type"] = per_type
    out["sens"] = sens

    # зависимость от L (80..120 шаг 5)
    Ls = list(range(80, 121, 5))
    curve = [dict(L=l, **case(l, lengths, demand)) for l in Ls]
    out["curve"] = curve

    # ---- графики -----------------------------------------------------------
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    # (а) схемы раскроя: 3 листа FFD и 3 листа оптимального плана
    opt_sheets, opt_lab = [], []
    for p, cnt in sorted(out["opt_plan"], key=lambda t: -t[1])[:3]:
        s_ = []
        for n, l in zip(p, lengths):
            s_ += [l] * n
        opt_sheets.append(s_)
        opt_lab.append(f"×{cnt}")
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6))
    pick = [0, 6, len(sh) - 1]                       # листы 1, 7 и последний
    draw_layout(axes[0], [sh[i] for i in pick], L,
                "FFD: листы №1, №7 и последний (№25)",
                labels=[f"№{i + 1}" for i in pick])
    draw_layout(axes[1], opt_sheets, L,
                "Оптимальный план (ЦЛП): три самые частые схемы",
                labels=opt_lab)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[l]) for l in lengths]
    fig.legend(handles, [f"деталь {l}" for l in lengths], loc="lower center",
               ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{save_dir}/fig_layouts.png", dpi=160)
    plt.close(fig)

    # (б) столбчатая диаграмма: листы и отходы для трёх подходов
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = ["Нижняя\nграница", "FFD", "Перебор"]
    s = out["sub"]
    sub_vals = [s["lb_int"], s["ffd_sheets"], s["bf_sheets"]]
    sub_w = [s["lb_waste"], s["ffd_waste"], s["bf_waste"]]
    full_vals = [out["lb_int"], out["ffd_sheets"], out["opt_sheets"]]
    full_w = [out["lb_int"] * L - out["total_len"], out["ffd_waste"],
              out["opt_waste"]]
    for ax, vals, wst, ttl in ((axes[0], sub_vals, sub_w,
                                "Подмножество (типы 1–3)"),
                               (axes[1], full_vals, full_w,
                                "Полная задача (5 типов)")):
        xs = np.arange(3)
        b1 = ax.bar(xs - 0.2, vals, 0.4, label="Число листов", color="#4C78A8")
        ax2 = ax.twinx()
        b2 = ax2.bar(xs + 0.2, wst, 0.4, label="Отходы, ед.", color="#F58518")
        for r, v in zip(b1, vals):
            ax.text(r.get_x() + r.get_width() / 2, v, str(v), ha="center",
                    va="bottom")
        for r, v in zip(b2, wst):
            ax2.text(r.get_x() + r.get_width() / 2, v, f"{v:g}", ha="center",
                     va="bottom")
        ax.set_xticks(xs)
        ax.set_xticklabels(names)
        ax.set_ylim(0, max(vals) * 1.25)
        ax2.set_ylim(0, max(wst) * 1.6 + 1)
        ax.set_ylabel("Листов")
        ax2.set_ylabel("Отходы, ед.")
        ax.set_title(ttl)
        ax.legend([b1, b2], ["Число листов", "Отходы, ед."], loc="upper left",
                  fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/fig_bars.png", dpi=160)
    plt.close(fig)

    # (в) доля отходов от длины листа
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(Ls, [c["ffd_pct"] for c in curve], "o-", label="FFD")
    ax.plot(Ls, [c["opt_pct"] for c in curve], "s--", label="Оптимум (ЦЛП)")
    ax.plot(Ls, [c["lb_pct"] for c in curve], "^:", label="Нижняя граница")
    for c in curve:
        ax.annotate(f"{c['ffd']}", (c["L"], c["ffd_pct"]),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=8)
    ax.set_xlabel("Длина листа L")
    ax.set_ylabel("Доля отходов, %")
    ax.set_title("Доля отходов в зависимости от L (подписи – число листов FFD)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{save_dir}/fig_waste_vs_L.png", dpi=160)
    plt.close(fig)

    with open(f"{save_dir}/results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    return out


if __name__ == "__main__":
    res = run_all(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(json.dumps({k: res[k] for k in
                      ("total_len", "lb_real", "lb_int", "lp_val",
                       "ffd_sheets", "ffd_waste", "opt_sheets", "opt_waste")},
                     ensure_ascii=False, indent=1))
