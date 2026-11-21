import itertools
import json
import math
import sys
from collections import Counter
from functools import lru_cache
from typing import List, Tuple

import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

# +----------------------------------------------------------------+
# |                                                                |
# |                        Вводные данные                          |
# |                                                                |
# +----------------------------------------------------------------+

EPS = 1e-9

L0 = 100
LENGTHS = [45, 35, 28, 20, 12]
DEMAND = [12, 15, 18, 22, 30]

# +----------------------------------------------------------------+
# |                                                                |
# |                        Utility функции                         |
# |                                                                |
# +----------------------------------------------------------------+

def total_length(lengths: List[int], demand: List[int]) -> int:
    """Суммарная длина всех нужных деталей"""
    return sum((l * b) for (l, b) in zip(lengths, demand));

def waste_stats(N: int, L: int, lengths: List[int], demand: List[int]) -> Tuple[int, float]:
    """Отходы для N листов длины L: (единиц отходов, доля от всего материала N * L в %)"""
    total = total_length(lengths, demand);
    num = N * L - total; # единиц отходов
    return (num, num * 100 / (N * L)); # в %

# +----------------------------------------------------------------+
# |                                                                |
# |                        Нижняя граница                          |
# |                                                                |
# +----------------------------------------------------------------+

def lower_bound_simple(L: int, lengths: List[int], demand: List[int], w=0):
    """
        Нижняя граница по суммарной длине: ceil(sum(l_i * b_i) / L) - не отходит от формулы.
        С пропилом w деталь занимает l + w, а лист L + w
    """
    s = sum((l + w) * b for l, b in zip(lengths, demand));
    N = s / (L + w);
    return (N, math.ceil(N - EPS))

def generate_patterns(L: int, lengths: List[int], w=0, maximal_only=False):
    """
        Все допустимые схемы раскроя одного листа.
        Схема - вектор (n_1,...,n_m), где n_i – число деталей типа i на листе,
        и  sum n_i*(l_i + w) <= L + w.  Пустая схема исключена.
    """

    # Количество элементов 
    m = len(lengths)

    # Длины деталей с учетом реза пилы
    lw_lengths = [l + w for l in lengths]
    
    # Длина листа с учетом пилы (пила может быть за пределами листа) 
    cap = L + w

    # Массив векторов (схем)
    patterns: List[Tuple[int]] = []

    def rec(i: int, remaining: int, cur: Tuple[int]):
        # Встретили последний элемент - добавляем вектор схемы в patterns
        if i == m:
            if any(cur):
                patterns.append(tuple(cur))
            return

        # Вычисляем, сколько деталей текущего типа сможем вместить в лист
        for k in range(remaining // lw_lengths[i] + 1):
            cur.append(k)
            rec(i + 1, remaining - k * lw_lengths[i], cur)
            cur.pop()

    rec(0, cap, [])

    if maximal_only:
        def used(p: List[int]):
            return sum(n * e for n, e in zip(p, lw_lengths))

        # Фильтруем схемы.
        # Если в схему можно уместить еще одну деталь, отбрасываем ее
        patterns = [p for p in patterns if all(used(p) + e > cap for e in lw_lengths)]
    
    return patterns

def lp_relaxation(L, lengths, demand, w=0):
    """
        Минимизировать - sum(x_p)
        
        Ограничение - итоговое количество деталей больше или равно нужному
            - sum(n_ip * x_p) >= b_i

        Число листов, раскроенных по каждой выбранной схеме >= 0
            - x_p >= 0.
        
        Возвращает (оптимум LP, число схем, решение x, двойственные оценки)
    """
    pats = generate_patterns(L, lengths, w, maximal_only=True)

    # Матрица из a_ip - из элементов,
    # отражающих количество деталей типа i в схеме p 
    A = np.array(pats, dtype=float).T

    c = np.ones(A.shape[1])
    
    res = linprog(
                c,
                # linprog принимает только ограничения вида "<=",
                # поэтому для ограничения ">=" меняем знаки для A_ub и b_ub
                A_ub=-A,
                b_ub=-np.array(demand, dtype=float),
                bounds=(0, None),
                # Симплекс-подобный решатель
                method="highs"
            )
    
    duals = -res.ineqlin.marginals # "цены" деталей ("цена" = часть листа, что занимает деталь)
    
    return (res.fun, len(pats), pats, res.x, duals)

# +----------------------------------------------------------------+
# |                                                                |
# |                           Раскрой                              |
# |                                                                |
# +----------------------------------------------------------------+

def ffd(L: int, lengths: List[int], demand: List[int], w=0):
    """
        Первый подходящий в порядке убывания.
        Возвращает список листов; лист – список длин деталей (в порядке укладки).
    """
    items = []
    for (l, b) in zip(lengths, demand):
        items += [l] * b
    
    items.sort(reverse=True)

    # sheets - состав деталей на листсе
    # free - свободные части для каждого листа
    # free[j] - свободная часть листа j
    sheets, free = [], []
    
    for it in items:
        need = it + w
        for j in range(len(sheets)):
            if free[j] >= need - EPS:
                sheets[j].append(it)
                free[j] -= need
                break

        # Если не смогли найти места на открытых листах, берем новый
        else:
            sheets.append([it])
            free.append(L + w - need)

    return sheets

def brute_force_cover(L: int, lengths: List[int], demand: List[int], w=0):
    """
        Точный перебор: минимум листов, покрывающих спрос,
        как комбинация схем раскроя
    """

    pats = generate_patterns(L, lengths, w, maximal_only=True)

    # чем больше полотна листа использует схема, тем раньше её пробуем
    eff = [l + w for l in lengths]

    # сортируем по убыванию от использованного полотна
    pats.sort(key=lambda p: -sum(n * e for n, e in zip(p, eff)))

    # включаем мемоизацию результатов
    @lru_cache(maxsize=None)
    def best(rem: Tuple[int]):
        # если не осталось, возвращаем 0
        if not any(rem):
            return (0, ())
        
        top = None
        
        # Бегаем по паттернам
        # пытаемся найти лучшую схему
        for p in pats:
            # получаем новую схему
            new = tuple(max(0, r - n) for (r, n) in zip(rem, p))

            # если получилась такая же, не перезапускаем рекурсию
            if new == rem:
                continue

            count, path = best(new)
            cand = (count + 1, (p,) + path)

            if top is None or cand[0] < top[0]:
                top = cand

        return top

    return (best(tuple(demand)), pats)

def exact_ilp(L, lengths, demand, w=0):
    """
        Точное целочисленное решение (HiGHS) на множестве максимальных схем.
    """
    
    pats = generate_patterns(L, lengths, w, maximal_only=True)
    
    A = np.array(pats, dtype=float).T
    P = A.shape[1]
    
    res = milp(
            c=np.ones(P),
            constraints=LinearConstraint(A,
                                         lb=np.array(demand, float),
                                         ub=np.inf
                                        ),
            integrality=np.ones(P),
            bounds=Bounds(0, np.inf)
        )
    
    x = np.round(res.x).astype(int)
    
    return (int(round(res.fun)), pats, x)

# Рост спроса в экспериментах чувствительности (+20 %)
DEMAND_GROWTH = 1.2

# Значения L для зависимости отходов от длины листа
L_RANGE = list(range(80, 121, 5))

def solve_case(L: int, lengths: List[int], demand: List[int], w=0) -> dict:
    """
        Все методы для одного набора данных:
        нижняя граница, LP-релаксация, FFD и точный оптимум (ЦЛП)
    """
    lb_real, lb = lower_bound_simple(L, lengths, demand, w)
    lp = lp_relaxation(L, lengths, demand, w)[0]
    n_ffd = len(ffd(L, lengths, demand, w))
    n_opt = exact_ilp(L, lengths, demand, w)[0]

    res = {
        "total": total_length(lengths, demand),
        "lb_real": lb_real,
        "lp": float(lp),
    }

    # Для каждого метода: число листов, отходы в единицах и в %
    for (name, n) in (("lb", lb), ("ffd", n_ffd), ("opt", n_opt)):
        waste, pct = waste_stats(n, L, lengths, demand)
        res[name] = n
        res[name + "_waste"] = waste
        res[name + "_pct"] = pct

    return res

def run_all(sub_n: int = 3) -> dict:
    L, lengths, demand = L0, LENGTHS, DEMAND
    out = {}

    # Исходные данные
    out["input"] = {
        "L": L,
        "lengths": lengths,
        "demand": demand,
        "type_total_lengths": [l * b for (l, b) in zip(lengths, demand)],
        "total_length": total_length(lengths, demand),
        "total_pieces": sum(demand),
    }

    # Нижняя граница
    lb_real, lb = lower_bound_simple(L, lengths, demand)
    lb_waste, lb_pct = waste_stats(lb, L, lengths, demand)
    lp_val, n_max, lp_pats, lp_x, duals = lp_relaxation(L, lengths, demand)

    out["lower_bound"] = {
        "simple_real": lb_real,
        "simple": lb,
        "simple_waste": lb_waste,
        "simple_pct": lb_pct,
        "lp": float(lp_val),
        "lp_ceil": math.ceil(lp_val - EPS),
        "n_patterns_all": len(generate_patterns(L, lengths)),
        "n_patterns_maximal": n_max,
        "lp_duals": [float(d) for d in duals],
        # ненулевые компоненты решения релаксации: (схема, x_p)
        "lp_solution": [(p, float(x)) for (p, x) in zip(lp_pats, lp_x) if x > EPS],
    }

    # FFD
    sheets = ffd(L, lengths, demand)
    ffd_waste, ffd_pct = waste_stats(len(sheets), L, lengths, demand)

    # Состав листов (набор деталей, сколько таких листов, отход на листе)
    comp = Counter(tuple(s) for s in sheets)
    composition = []
    for (s, cnt) in comp.items():
        w_sheet = L - sum(s)
        composition.append({
            "sheet": list(s),
            "count": cnt,
            "waste_per_sheet": w_sheet,
            "waste_total": w_sheet * cnt,
        })
    composition.sort(key=lambda c: (-c["sheet"][0], -c["count"]))

    # Отходы на листах, где только детали из первых трех типов (45, 35, 28)
    large = set(lengths[:3])
    large_waste = sum(L - sum(s) for s in sheets if set(s) <= large)

    out["ffd"] = {
        "sheets": len(sheets),
        "waste": ffd_waste,
        "pct": ffd_pct,
        "layout": sheets,
        "composition": composition,
        "waste_on_large_only_sheets": large_waste,
    }

    # Подзадача (первые sub_n типов): полный перебор
    sub_l, sub_b = lengths[:sub_n], demand[:sub_n]
    (bf_count, bf_path), sub_pats = brute_force_cover(L, sub_l, sub_b)

    sub_lb_real, sub_lb = lower_bound_simple(L, sub_l, sub_b)
    sub_lp = lp_relaxation(L, sub_l, sub_b)[0]
    sub_ffd = len(ffd(L, sub_l, sub_b))

    sub = {
        "lengths": sub_l,
        "demand": sub_b,
        "total_length": total_length(sub_l, sub_b),
        "total_pieces": sum(sub_b),
        "n_patterns_all": len(generate_patterns(L, sub_l)),
        "n_patterns_maximal": len(sub_pats),
        # максимальные схемы по убыванию занятой длины (схема, отход)
        "patterns": [(p, L - sum(n * l for (n, l) in zip(p, sub_l))) for p in sub_pats],
        "lb_real": sub_lb_real,
        "lp": float(sub_lp),
    }
    for (name, n) in (("lb", sub_lb), ("ffd", sub_ffd), ("bf", bf_count)):
        waste, pct = waste_stats(n, L, sub_l, sub_b)
        sub[name] = n
        sub[name + "_waste"] = waste
        sub[name + "_pct"] = pct

    # План перебора (схема, число листов, отход на листе)
    sub["bf_plan"] = [
        (p, cnt, L - sum(n * l for (n, l) in zip(p, sub_l)))
        for (p, cnt) in Counter(bf_path).items()
    ]
    out["sub"] = sub

    # ---- Точный оптимум полной задачи (ЦЛП) ----
    n_opt, opt_pats, opt_x = exact_ilp(L, lengths, demand)
    opt_waste, opt_pct = waste_stats(n_opt, L, lengths, demand)

    # План: (схема, число листов, отход на листе), по убыванию числа листов
    plan = [
        (p, int(k), L - sum(n * l for (n, l) in zip(p, lengths)))
        for (p, k) in zip(opt_pats, opt_x) if k > 0
    ]
    plan.sort(key=lambda t: -t[1])

    # Обрезки внутри листов и лишние детали (в плане они не вырезаются)
    trim = sum(cnt * w_sheet for (_, cnt, w_sheet) in plan)

    # Десять лучших максимальных схем (по возрастанию отхода)
    good = sorted((L - sum(n * l for (n, l) in zip(p, lengths)), p) for p in lp_pats)

    out["opt"] = {
        "sheets": n_opt,
        "waste": opt_waste,
        "pct": opt_pct,
        "plan": plan,
        "trim_waste": trim,
        "surplus_waste": opt_waste - trim,
        "best_patterns": [(p, w_sheet) for (w_sheet, p) in good[:10]],
    }

    sens = {}

    # Базовый случай и другие длины листа
    sens["base"] = solve_case(L, lengths, demand)
    sens["L90"] = solve_case(90, lengths, demand)
    sens["L110"] = solve_case(110, lengths, demand)

    # Спрос на типы 1 и 5 вырос на 20 %
    d20 = list(demand)
    for i in (0, 4):
        d20[i] = math.ceil(demand[i] * DEMAND_GROWTH - EPS)
    sens["demand20"] = solve_case(L, lengths, d20)
    sens["demand20"]["demand"] = d20

    # Зазор на рез w = 1
    sens["kerf1"] = solve_case(L, lengths, demand, 1)

    # Рост спроса на 20 % по одному типу
    per_type = []
    for i in range(len(lengths)):
        d = list(demand)
        d[i] = math.ceil(d[i] * DEMAND_GROWTH - EPS)
        per_type.append({"type": i + 1, "demand": d[i], **solve_case(L, lengths, d)})
    sens["per_type"] = per_type

    # Зависимость от длины листа
    sens["curve"] = [{"L": l, **solve_case(l, lengths, demand)} for l in L_RANGE]

    out["sensitivity"] = sens

    # Отход при раскрое листа деталями одного типа
    out["single_type"] = [
        {
            "type": i + 1,
            "length": l,
            "fits": L // l,
            "used": (L // l) * l,
            "waste": L - (L // l) * l,
        }
        for (i, l) in enumerate(lengths)
    ]

    return out

if __name__ == "__main__":
    result = run_all()
    print(json.dumps(result, ensure_ascii=False, indent=2))