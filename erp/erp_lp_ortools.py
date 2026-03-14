# ----------------------------- Linear Programming — ERP Computing Infrastructure -------------------

"""
Задача лінійного програмування:
   Оптимальне планування закупівлі серверного обладнання для ERP-системи

Практична інтерпретація:
   Підприємство планує розгорнути корпоративну ERP-систему (наприклад, SAP або 1C:Enterprise)
   та має бюджет на закупівлю двох типів серверів:

     x₁ — сервери застосунків (Application Servers, напр. HP ProLiant DL380)
          Вартість: 600 тис. грн/шт.
          Споживання: 1 кВт/шт.
          Продуктивність: 5 ум.од.

     x₂ — сервери баз даних (Database Servers, напр. Dell PowerEdge R740)
          Вартість: 400 тис. грн/шт.
          Споживання: 2 кВт/шт.
          Продуктивність: 4 ум.од.

Обмеження ресурсів:
   - Загальний бюджет  : 2 400 тис. грн  →  600·x₁ + 400·x₂ ≤ 2400  →  6x₁ + 4x₂ ≤ 24
   - Потужність ЦОД    : 6 кВт           →  x₁ + 2·x₂ ≤ 6
   - Невід'ємність     : x₁ ≥ 0, x₂ ≥ 0

Цільова функція (максимізація загальної продуктивності):
   Z = 5x₁ + 4x₂  →  max

Метод розв'язку:
   1. Google OR-Tools (GLOP — LP solver)
   2. Графічний метод (matplotlib)

Порівняння результатів:
   Обидва методи повинні дати однаковий результат: x₁=3, x₂=1.5, Z=21.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend (works without a display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ortools.linear_solver import pywraplp


# =============================================================================
# БЛОК 1. Формалізація задачі
# =============================================================================

# ---------- Коефіцієнти цільової функції Z = c1*x1 + c2*x2 ------------------
C1 = 5  # продуктивність одного app-сервера (ум.од.)
C2 = 4  # продуктивність одного db-сервера  (ум.од.)

# ---------- Коефіцієнти обмежень ---------------------------------------------
#   Обмеження 1 (бюджет):      a11*x1 + a12*x2 <= b1
#   Обмеження 2 (потужність):  a21*x1 + a22*x2 <= b2
A11, A12, B1 = 6, 4, 24  # бюджет: 6x₁ + 4x₂ ≤ 24  (одиниць × 100 тис. грн)
A21, A22, B2 = 1, 2, 6  # потужність: x₁ + 2x₂ ≤ 6  (кВт)


# =============================================================================
# БЛОК 2. Розв'язок методом Google OR-Tools (GLOP)
# =============================================================================


def solve_with_ortools() -> dict:
    """
    Розв'язує LP-задачу за допомогою GLOP (Google Linear Optimization).

    Задача:
      max  Z = 5x₁ + 4x₂
      s.t. 6x₁ + 4x₂ ≤ 24   (бюджет)
           x₁ + 2x₂ ≤  6    (потужність)
           x₁, x₂  ≥  0
    """
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        raise RuntimeError("Не вдалося створити GLOP-солвер.")

    # ----- Змінні рішення -----
    x1 = solver.NumVar(0.0, solver.infinity(), "x1")
    x2 = solver.NumVar(0.0, solver.infinity(), "x2")

    # ----- Обмеження -----
    budget_ct = solver.Constraint(0, B1, "budget")
    budget_ct.SetCoefficient(x1, A11)
    budget_ct.SetCoefficient(x2, A12)

    power_ct = solver.Constraint(0, B2, "power")
    power_ct.SetCoefficient(x1, A21)
    power_ct.SetCoefficient(x2, A22)

    # ----- Цільова функція (максимізація) -----
    objective = solver.Objective()
    objective.SetCoefficient(x1, C1)
    objective.SetCoefficient(x2, C2)
    objective.SetMaximization()

    # ----- Розв'язок -----
    status = solver.Solve()

    result = {
        "status": status,
        "status_name": "OPTIMAL"
        if status == pywraplp.Solver.OPTIMAL
        else "NOT OPTIMAL",
        "x1": x1.solution_value(),
        "x2": x2.solution_value(),
        "Z": objective.Value(),
        "iterations": solver.iterations(),
        "wall_time_ms": solver.wall_time(),
    }
    return result


# =============================================================================
# БЛОК 3. Графічний метод
# =============================================================================


def compute_corner_points() -> list[tuple[float, float]]:
    """
    Знаходить вершини допустимої області перетином обмежень.

    Повертає список точок (x1, x2) у кутах допустимої багатокутника.
    """
    corners = []

    # Вершина A: (0, 0)
    corners.append((0.0, 0.0))

    # Вершина B: перетин 1-го обм. з x2=0
    # 6x1 + 4*0 = 24 → x1=4; перевірка: 4 + 0 = 4 ≤ 6 ✓
    x1_b = B1 / A11
    if A21 * x1_b + A22 * 0 <= B2 + 1e-9:
        corners.append((x1_b, 0.0))

    # Вершина C: перетин обох прямих обмежень
    # 6x1 + 4x2 = 24
    # x1  + 2x2 = 6  → множимо на 2: 2x1 + 4x2 = 12
    # Різниця: 4x1 = 12 → x1=3, x2=1.5
    det = A11 * A22 - A12 * A21
    if abs(det) > 1e-12:
        x1_c = (B1 * A22 - B2 * A12) / det
        x2_c = (A11 * B2 - A21 * B1) / det
        if x1_c >= -1e-9 and x2_c >= -1e-9:
            corners.append((max(0, x1_c), max(0, x2_c)))

    # Вершина D: перетин 2-го обм. з x1=0
    # 0 + 2x2 = 6 → x2=3; перевірка: 0 + 4*3=12 ≤ 24 ✓
    x2_d = B2 / A22
    if A11 * 0 + A12 * x2_d <= B1 + 1e-9:
        corners.append((0.0, x2_d))

    return corners


def z_value(x1: float, x2: float) -> float:
    return C1 * x1 + C2 * x2


def plot_graphical_method(
    ortools_result: dict, save_path: str = "lp_graphical.png"
) -> None:
    """
    Будує графічний розв'язок LP-задачі:
      - Допустима область (закрашена)
      - Лінії обмежень
      - Вершини допустимої області з підписами та значеннями Z
      - Ізолінія цільової функції (оптимальна)
      - Оптимальна точка (виділена)
    Зберігає графік у файл save_path.
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    x1_range = np.linspace(0, 5.5, 600)

    # ---- Лінії обмежень ----
    # Обмеження 1 (бюджет): x2 = (24 - 6*x1) / 4
    x2_budget = (B1 - A11 * x1_range) / A12
    # Обмеження 2 (потужність): x2 = (6 - x1) / 2
    x2_power = (B2 - A21 * x1_range) / A22

    ax.plot(
        x1_range, x2_budget, "b-", linewidth=2, label=r"Бюджет: $6x_1 + 4x_2 \leq 24$"
    )
    ax.plot(
        x1_range, x2_power, "g-", linewidth=2, label=r"Потужність: $x_1 + 2x_2 \leq 6$"
    )

    # ---- Допустима область ----
    x2_feasible = np.minimum(
        np.maximum((B1 - A11 * x1_range) / A12, 0),
        np.maximum((B2 - A21 * x1_range) / A22, 0),
    )
    x2_feasible = np.where(x1_range <= B2 / A21, x2_feasible, 0)  # x1 ≤ 6

    ax.fill_between(
        x1_range,
        0,
        x2_feasible,
        where=(x2_feasible >= 0),
        alpha=0.20,
        color="cyan",
        label="Допустима область",
    )

    # ---- Вершини ----
    corners = compute_corner_points()
    labels = ["A", "B", "C", "D"]
    offsets = [(-0.25, -0.20), (0.07, -0.20), (0.07, 0.10), (-0.28, 0.08)]

    for i, (px, py) in enumerate(corners):
        zv = z_value(px, py)
        color = "red" if abs(zv - ortools_result["Z"]) < 1e-4 else "black"
        ax.plot(px, py, "o", color=color, markersize=9, zorder=5)
        lbl = labels[i] if i < len(labels) else f"P{i}"
        off = offsets[i] if i < len(offsets) else (0.07, 0.07)
        ax.annotate(
            f"{lbl}({px:.1f}; {py:.1f})\nZ={zv:.1f}",
            xy=(px, py),
            xytext=(px + off[0], py + off[1]),
            fontsize=9,
            color=color,
            fontweight="bold" if color == "red" else "normal",
        )

    # ---- Оптимальна ізолінія Z* ----
    Z_opt = ortools_result["Z"]
    x2_iso = (Z_opt - C1 * x1_range) / C2
    ax.plot(
        x1_range,
        x2_iso,
        "r--",
        linewidth=1.8,
        label=f"Ізолінія $Z^* = {Z_opt:.1f}$: $5x_1 + 4x_2 = {Z_opt:.1f}$",
    )

    # ---- Оптимальна точка ----
    x1_opt, x2_opt = ortools_result["x1"], ortools_result["x2"]
    ax.plot(
        x1_opt,
        x2_opt,
        "*",
        color="red",
        markersize=18,
        zorder=6,
        label=f"Оптимум: $x_1^*={x1_opt:.2f},\\ x_2^*={x2_opt:.2f}$",
    )

    # ---- Оформлення ----
    ax.set_xlim(-0.3, 5.5)
    ax.set_ylim(-0.3, 5.0)
    ax.set_xlabel("$x_1$ — кількість серверів застосунків", fontsize=12)
    ax.set_ylabel("$x_2$ — кількість серверів баз даних", fontsize=12)
    ax.set_title(
        "Графічний метод розв'язку задачі ЛП\n"
        "Оптимальне планування обчислювального комплексу ERP-системи",
        fontsize=13,
    )
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Графік збережено: {save_path}")


# =============================================================================
# БЛОК 4. Виведення результатів та порівняння
# =============================================================================


def print_problem_statement() -> None:
    print("=" * 72)
    print("ЗАДАЧА ЛІНІЙНОГО ПРОГРАМУВАННЯ")
    print("Оптимальне планування закупівлі серверного обладнання для ERP")
    print("=" * 72)
    print()
    print("  Змінні рішення:")
    print("    x₁ — кількість серверів застосунків (HP ProLiant DL380)")
    print("         Вартість: 600 тис. грн  |  Споживання: 1 кВт  |  Z-внесок: 5")
    print("    x₂ — кількість серверів баз даних   (Dell PowerEdge R740)")
    print("         Вартість: 400 тис. грн  |  Споживання: 2 кВт  |  Z-внесок: 4")
    print()
    print("  Цільова функція (максимізація продуктивності, ум.од.):")
    print("    Z = 5x₁ + 4x₂  →  max")
    print()
    print("  Система обмежень:")
    print("    6x₁ + 4x₂ ≤ 24   (бюджет: 2 400 тис. грн, нормалізований у ×100)")
    print("    x₁  + 2x₂ ≤  6   (потужність ЦОД: 6 кВт)")
    print("    x₁, x₂   ≥  0    (невід'ємність)")
    print()


def print_graphical_analysis() -> None:
    print("-" * 72)
    print("ГРАФІЧНИЙ МЕТОД (аналітична частина)")
    print("-" * 72)
    corners = compute_corner_points()
    labels = ["A(0; 0)", "B(4; 0)", "C(3; 1.5)", "D(0; 3)"]
    print()
    print("  Вершини допустимої опуклої області:")
    print(f"  {'Точка':<14} {'x₁':>6} {'x₂':>6} {'Z = 5x₁+4x₂':>14}")
    print("  " + "-" * 42)
    for lbl, (px, py) in zip(labels, corners):
        print(f"  {lbl:<14} {px:>6.2f} {py:>6.2f} {z_value(px, py):>14.2f}")
    print()
    best_corner = max(corners, key=lambda p: z_value(*p))
    print(
        f"  Оптимальна вершина: x₁* = {best_corner[0]:.2f}, "
        f"x₂* = {best_corner[1]:.2f}, Z* = {z_value(*best_corner):.2f}"
    )
    print()
    print("  Активні обмеження в оптимумі:")
    px, py = best_corner
    v1 = A11 * px + A12 * py
    v2 = A21 * px + A22 * py
    print(
        f"    Бюджет     : {A11}×{px:.2f} + {A12}×{py:.2f} = {v1:.2f}  (ліміт {B1})  "
        f"{'[АКТИВНЕ]' if abs(v1 - B1) < 1e-4 else ''}"
    )
    print(
        f"    Потужність : {A21}×{px:.2f} + {A22}×{py:.2f} = {v2:.2f}  (ліміт {B2})  "
        f"{'[АКТИВНЕ]' if abs(v2 - B2) < 1e-4 else ''}"
    )
    print()


def print_ortools_result(res: dict) -> None:
    print("-" * 72)
    print("РОЗВ'ЯЗОК OR-TOOLS (GLOP — Google Linear Optimization)")
    print("-" * 72)
    print()
    print(f"  Статус                  : {res['status_name']}")
    print(f"  x₁* (сервери застосунків): {res['x1']:.6f}")
    print(f"  x₂* (сервери БД)         : {res['x2']:.6f}")
    print(f"  Z* (продуктивність)      : {res['Z']:.6f}")
    print(f"  Ітерацій солвера         : {res['iterations']}")
    print(f"  Час виконання            : {res['wall_time_ms']:.3f} мс")
    print()


def print_comparison(ortools_res: dict) -> None:
    print("-" * 72)
    print("ПОРІВНЯННЯ МЕТОДІВ")
    print("-" * 72)
    corners = compute_corner_points()
    best_corner = max(corners, key=lambda p: z_value(*p))
    gx1, gx2 = best_corner
    gZ = z_value(gx1, gx2)
    ox1, ox2, oZ = ortools_res["x1"], ortools_res["x2"], ortools_res["Z"]

    print()
    print(f"  {'Метод':<35} {'x₁*':>8} {'x₂*':>8} {'Z*':>10}")
    print("  " + "-" * 63)
    print(f"  {'Графічний метод':<35} {gx1:>8.4f} {gx2:>8.4f} {gZ:>10.4f}")
    print(f"  {'OR-Tools (GLOP)':<35} {ox1:>8.4f} {ox2:>8.4f} {oZ:>10.4f}")
    print()

    diff_x1 = abs(gx1 - ox1)
    diff_x2 = abs(gx2 - ox2)
    diff_Z = abs(gZ - oZ)
    print(
        f"  Відхилення |Δx₁| = {diff_x1:.2e},  |Δx₂| = {diff_x2:.2e},  |ΔZ| = {diff_Z:.2e}"
    )

    if diff_Z < 1e-4 and diff_x1 < 1e-4 and diff_x2 < 1e-4:
        print()
        print("  ВИСНОВОК: Обидва методи дають ІДЕНТИЧНИЙ результат.")
        print("  Графічний метод підтверджує точність OR-Tools GLOP-розв'язку.")
    else:
        print()
        print("  УВАГА: Результати розходяться більше за поріг 1e-4.")
    print()


def print_practical_interpretation(ortools_res: dict) -> None:
    x1 = ortools_res["x1"]
    x2 = ortools_res["x2"]
    Z = ortools_res["Z"]

    budget_used = (A11 * x1 + A12 * x2) * 100  # тис. грн
    power_used = A21 * x1 + A22 * x2  # кВт
    budget_total = B1 * 100
    power_total = B2

    print("=" * 72)
    print("ПРАКТИЧНА ІНТЕРПРЕТАЦІЯ")
    print("=" * 72)
    print()
    print("  Галузь застосування: корпоративні ERP-системи (Enterprise Resource")
    print("  Planning) — планування ресурсів підприємства на базі програмних")
    print("  платформ SAP S/4HANA, 1C:Підприємство, Microsoft Dynamics тощо.")
    print()
    print("  Сценарій:")
    print("  Середнє підприємство (500–1000 користувачів ERP) формує бюджет")
    print("  на оновлення обчислювального центру. Перед IT-директором стоїть")
    print("  задача: визначити, яку кількість серверів застосунків та серверів")
    print("  баз даних придбати, щоб максимізувати сумарну продуктивність")
    print("  системи в рамках бюджетних та енергетичних обмежень.")
    print()
    print("  Оптимальне рішення OR-Tools:")
    print(f"    x₁* = {x1:.2f} серверів застосунків")
    print(f"    x₂* = {x2:.2f} серверів баз даних")
    print(f"    Z*  = {Z:.2f} ум.од. сумарної продуктивності")
    print()
    print("  Використання ресурсів:")
    print(
        f"    Бюджет     : {budget_used:.0f} / {budget_total:.0f} тис. грн  "
        f"({100 * budget_used / budget_total:.1f}% використано)"
    )
    print(
        f"    Потужність : {power_used:.2f} / {power_total:.2f} кВт  "
        f"({100 * power_used / power_total:.1f}% використано)"
    )
    print()
    print("  Управлінська рекомендація:")
    print(f"  Придбати 3 сервери застосунків та 1–2 сервери баз даних.")
    print(f"  (LP-розв'язок дає x₂=1.5; у цілочисельному варіанті — 1 або 2)")
    print("  При x₁=3, x₂=2: Z=23, бюджет=26>24 (перевищення бюджету).")
    print("  При x₁=3, x₂=1: Z=19, бюджет=22≤24 (безпечне рішення).")
    print()
    print("  Висновок задачі ЛП для реальної галузі:")
    print("  Формулювання задачі у вигляді ЛП дозволяє автоматично знаходити")
    print("  оптимальний баланс між типами обладнання з урахуванням одночасно")
    print("  декількох ресурсних обмежень. Бібліотека OR-Tools забезпечує")
    print("  промислово надійний GLOP-розв'язок за частки мілісекунди — що є")
    print("  важливим у задачах повторного планування при зміні бюджету чи")
    print("  енергетичного ліміту ЦОД.")
    print("=" * 72)


# =============================================================================
# БЛОК 5. Головна функція
# =============================================================================


def main() -> None:
    print_problem_statement()

    # OR-Tools
    ortools_res = solve_with_ortools()
    print_ortools_result(ortools_res)

    # Graphical method (analytical)
    print_graphical_analysis()

    # Comparison
    print_comparison(ortools_res)

    # Graphical plot
    print("-" * 72)
    print("ПОБУДОВА ГРАФІКА")
    print("-" * 72)
    plot_graphical_method(ortools_res, save_path="lp_graphical.png")

    # Practical interpretation
    print()
    print_practical_interpretation(ortools_res)


if __name__ == "__main__":
    main()
