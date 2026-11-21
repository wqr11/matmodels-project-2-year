clear; clc;

L0 = 100;
LENGTHS = [45 35 28 20 12];
DEMAND = [12 15 18 22 30];

OPT_PLAN = [1 1 0 1 0;
            0 0 2 1 2;
            0 0 1 0 6;
            0 1 0 2 2;
            0 1 1 0 3;
            0 1 1 1 1];
OPT_COUNTS = [12; 7; 2; 1; 1; 1];

L_RANGE = 80:5:120;
OPT_BY_L = [30 29 27 26 24 23 22 21 20];

function [waste, pct] = waste_stats(N, L, lengths, demand)
    waste = N * L - sum(lengths .* demand);
    pct = waste * 100 / (N * L);
endfunction

function sheet = pattern_to_sheet(p, lengths)
    sheet = [];
    for i = 1:length(lengths)
        sheet = [sheet, lengths(i) * ones(1, p(i))];
    end
endfunction

function N = lower_bound_simple(L, lengths, demand)
    N = ceil(sum(lengths .* demand) / L);
endfunction

function pats = enum_patterns(i, remaining, lengths)
    m = length(lengths);
    last = floor(remaining / lengths(i));

    if i == m then
        pats = (0:last)';
        return;
    end

    pats = [];
    for k = 0:last
        sub = enum_patterns(i + 1, remaining - k * lengths(i), lengths);
        pats = [pats; k * ones(size(sub, 1), 1), sub];
    end
endfunction

function pats = generate_patterns(L, lengths)
    pats = enum_patterns(1, L, lengths);
    used = pats * lengths';
    pats = pats(used + min(lengths) > L, :);
endfunction

function sheets = ffd(L, lengths, demand)
    items = [];
    for i = 1:length(lengths)
        items = [items, lengths(i) * ones(1, demand(i))];
    end
    items = gsort(items, "g", "d");

    sheets = list();
    free = [];
    n_sheets = 0;

    for it = items
        placed = %f;

        for j = 1:n_sheets
            if free(j) >= it then
                sheets(j) = [sheets(j), it];
                free(j) = free(j) - it;
                placed = %t;
                break;
            end
        end

        if ~placed then
            n_sheets = n_sheets + 1;
            sheets(n_sheets) = it;
            free(n_sheets) = L - it;
        end
    end
endfunction

function r = decode_state(idx, demand)
    m = length(demand);
    r = zeros(1, m);
    t = idx - 1;
    for i = 1:m
        r(i) = modulo(t, demand(i) + 1);
        t = floor(t / (demand(i) + 1));
    end
endfunction

function n_sheets = brute_force_cover(L, lengths, demand)
    // Перебор "снизу вверх"
	// dp(idx) - минимум листов для остатка спроса с номером idx
    pats = generate_patterns(L, lengths);
    P = size(pats, 1);
    m = length(demand);

    strides = ones(1, m);
    for i = 2:m
        strides(i) = strides(i - 1) * (demand(i - 1) + 1);
    end
    S = prod(demand + 1);

    dp = zeros(S, 1);
    for idx = 2:S
        r = decode_state(idx, demand);

        new = ones(P, 1) * r - pats;
        new(new < 0) = 0;
        nidx = 1 + new * strides';

        dp(idx) = min(dp(nidx(nidx <> idx))) + 1;
    end

    n_sheets = dp(S);
endfunction

function [cols, greyc] = make_palette()
    cols = [color(76, 120, 168), color(245, 133, 24), color(84, 162, 75), ..
            color(178, 121, 162), color(228, 87, 86)];
    greyc = color(221, 221, 221);
endfunction

function draw_box(x, y, w, h, c)
    xrect(x, y, w, h);
    e = gce();
    e.fill_mode = "on";
    e.background = c;
    e.foreground = color("black");
endfunction

function draw_label(xc, yc, s, fs, fc)
    xstring(xc, yc, s);
    t = gce();
    t.font_size = fs;
    t.font_foreground = fc;

    r = stringbox(t);
    wd = max(r(1, :)) - min(r(1, :));
    ht = max(r(2, :)) - min(r(2, :));
    t.data = [xc - wd / 2, yc - ht / 2];
endfunction

function draw_layout(rows, row_labels, ttl, L, lengths, with_legend)
    [cols, greyc] = make_palette();

    n_rows = length(rows);
    a = gca();

    y_min = -1.5 * (n_rows - 1) - 1.5;
    if with_legend then
        y_min = y_min - 1.4;
    end
    a.data_bounds = [0, y_min; L, 0.6];
    a.tight_limits = "on";

    for r = 1:n_rows
        top = -1.5 * (r - 1);
        pcs = rows(r);
        x = 0;

        for k = 1:length(pcs)
            l = pcs(k);
            draw_box(x, top, l, 1, cols(find(lengths == l)));
            draw_label(x + l / 2, top - 0.5, string(l), 3, color("white"));
            x = x + l;
        end

        rest = L - x;
        if rest > 0 then
            draw_box(x, top, rest, 1, greyc);
            if rest >= 12 then
                s = "отход " + string(rest);
            else
                s = string(rest);
            end
            draw_label(x + rest / 2, top - 0.5, s, 2, color("black"));
        end
    end

    loc = zeros(n_rows, 1);
    for r = 1:n_rows
        loc(r) = -1.5 * (r - 1) - 0.5;
    end
    a.y_ticks = tlist(["ticks", "locations", "labels"], loc, row_labels);

    if with_legend then
        ly = -1.5 * (n_rows - 1) - 1.9;
        for i = 1:length(lengths)
            x0 = 3 + (i - 1) * 19;
            draw_box(x0, ly, 5, 0.8, cols(i));
            xstring(x0 + 6.5, ly - 0.75, "деталь " + string(lengths(i)));
        end
    end

    xtitle(ttl, "Длина, ед.", "");
endfunction

function draw_bars(vals, names, ttl, y_label)
    y_max = max(vals) * 1.25;

    bar(1:3, vals, 0.5);
    a = gca();
    a.data_bounds = [0.4, 0; 3.6, y_max];
    a.tight_limits = "on";
    a.x_ticks = tlist(["ticks", "locations", "labels"], [1; 2; 3], names);

    for i = 1:3
        xstring(i - 0.1, vals(i) + y_max * 0.02, string(vals(i)));
    end

    xtitle(ttl, "", y_label);
endfunction

// Расчеты: полная задача и подзадача (типы 1-3)
n_lb = lower_bound_simple(L0, LENGTHS, DEMAND);
sheets = ffd(L0, LENGTHS, DEMAND);
n_ffd = length(sheets);
n_opt = sum(OPT_COUNTS);

sub_lengths = LENGTHS(1:3);
sub_demand = DEMAND(1:3);
n_sub_lb = lower_bound_simple(L0, sub_lengths, sub_demand);
n_sub_ffd = length(ffd(L0, sub_lengths, sub_demand));
n_sub_bf = brute_force_cover(L0, sub_lengths, sub_demand);

// Рисунок 1: схемы раскроя
rows_ffd = list(sheets(1), sheets(7), sheets(n_ffd));
labels_ffd = ["№1"; "№7"; "№" + string(n_ffd)];

[sorted_counts, order] = gsort(OPT_COUNTS, "g", "d");
rows_opt = list();
labels_opt = [];
for r = 1:3
    rows_opt($ + 1) = pattern_to_sheet(OPT_PLAN(order(r), :), LENGTHS);
    labels_opt = [labels_opt; "×" + string(sorted_counts(r))];
end

f1 = scf(1);
clf(f1);
f1.figure_size = [900, 650];

subplot(2, 1, 1);
draw_layout(rows_ffd, labels_ffd, "FFD: листы №1, №7 и последний", L0, LENGTHS, %f);

subplot(2, 1, 2);
draw_layout(rows_opt, labels_opt, "Оптимальный план (ЦЛП): три самые частые схемы", L0, LENGTHS, %t);

xs2png(f1, "scilab_fig_layouts.png");

// Рисунок 2: число листов и отходы
names = ["Нижняя граница"; "FFD"; "Перебор"];

sub_n = [n_sub_lb, n_sub_ffd, n_sub_bf];
full_n = [n_lb, n_ffd, n_opt];
sub_w = zeros(1, 3);
full_w = zeros(1, 3);
for i = 1:3
    sub_w(i) = waste_stats(sub_n(i), L0, sub_lengths, sub_demand);
    full_w(i) = waste_stats(full_n(i), L0, LENGTHS, DEMAND);
end

f2 = scf(2);
clf(f2);
f2.figure_size = [1000, 650];

subplot(2, 2, 1);
draw_bars(sub_n, names, "Типы 1-3: число листов", "Листов");
subplot(2, 2, 2);
draw_bars(full_n, names, "Полная задача: число листов", "Листов");
subplot(2, 2, 3);
draw_bars(sub_w, names, "Типы 1-3: отходы", "Отходы, ед.");
subplot(2, 2, 4);
draw_bars(full_w, names, "Полная задача: отходы", "Отходы, ед.");

xs2png(f2, "scilab_fig_bars.png");

// Рисунок 3: доля отходов в зависимости от L
n_L = length(L_RANGE);
pct_lb = zeros(1, n_L);
pct_ffd = zeros(1, n_L);
pct_opt = zeros(1, n_L);
n_ffd_L = zeros(1, n_L);

for i = 1:n_L
    L = L_RANGE(i);
    n_ffd_L(i) = length(ffd(L, LENGTHS, DEMAND));

    [tmp, pct_lb(i)] = waste_stats(lower_bound_simple(L, LENGTHS, DEMAND), L, LENGTHS, DEMAND);
    [tmp, pct_ffd(i)] = waste_stats(n_ffd_L(i), L, LENGTHS, DEMAND);
    [tmp, pct_opt(i)] = waste_stats(OPT_BY_L(i), L, LENGTHS, DEMAND);
end

f3 = scf(3);
clf(f3);
f3.figure_size = [850, 480];

plot(L_RANGE, pct_ffd, "b-o");
plot(L_RANGE, pct_opt, "r--s");
plot(L_RANGE, pct_lb, "k:^");

a = gca();
a.data_bounds = [78, 0; 122, 9];
a.tight_limits = "on";
xgrid(color("lightgray"));

for i = 1:n_L
    xstring(L_RANGE(i) - 0.6, pct_ffd(i) + 0.3, string(n_ffd_L(i)));
end

xtitle("Доля отходов в зависимости от L (подписи - число листов FFD)", ..
       "Длина листа L", "Доля отходов, %");
legend(["FFD"; "Оптимум (ЦЛП)"; "Нижняя граница"], -1);

xs2png(f3, "scilab_fig_waste_vs_L.png");
