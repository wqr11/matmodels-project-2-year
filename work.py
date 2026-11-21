import itertools
import json
import math
import sys
from functools import lru_cache

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
    return sum((l * b) for (i,b) in zip(lengths, demand));

def waste_stats(N: int, L: int, lengths: List[int], demand: List[int]) -> float:
    total = total_length(lengths, demand);
    num = N * L - total; # единиц отходов
    return num * 100 / total; # в %

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
    