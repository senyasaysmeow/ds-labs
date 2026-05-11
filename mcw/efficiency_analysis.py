import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# 1. Генерація даних
# =============================================================================

def generate_data() -> tuple[np.ndarray, list[str], list[str], list[str]]:
    np.random.seed(42)
    
    product_names = [f"Товар {i}" for i in range(1, 11)]
    criteria_names = [
        "Прибутковість (%)",
        "Обсяг продажів (шт)",
        "Відсоток браку (%)"
    ]
    criteria_type = ["мах", "мах", "мін"]
    
    # Створюємо матрицю значень: рядки = критерії, стовпці = товари
    # C1: Прибутковість (%) - максимізується
    c1 = np.random.uniform(10, 50, 10).round(2)
    # C2: Обсяг продажів (шт) - максимізується
    c2 = np.random.randint(100, 1000, 10).astype(float)
    # C3: Відсоток браку (%) - мінімізується
    c3 = np.random.uniform(0.5, 5.0, 10).round(2)
    
    matrix = np.vstack((c1, c2, c3))
    
    print("=" * 70)
    print("ВХІДНІ ДАНІ")
    print("=" * 70)
    print(f"  Кількість критеріїв : {len(criteria_names)}")
    print(f"  Кількість товарів   : {len(product_names)}")
    print(f"  Мінімізованих       : {criteria_type.count('мін')}")
    print(f"  Максимізованих      : {criteria_type.count('мах')}")
    print("=" * 70)
    
    # Вивід вихідних даних у вигляді таблиці
    df_raw = pd.DataFrame(matrix.T, columns=criteria_names, index=product_names)
    print("\nВИХІДНА МАТРИЦЯ:")
    print(df_raw.to_string())
    
    return matrix, criteria_names, product_names, criteria_type

# =============================================================================
# 2. Нормалізація
# =============================================================================

def normalize_matrix(matrix: np.ndarray, criteria_type: list[str]) -> np.ndarray:
    n_criteria, n_products = matrix.shape
    norm_matrix = np.zeros_like(matrix, dtype=float)

    for k in range(n_criteria):
        row = matrix[k, :]
        if criteria_type[k] == "мін":
            s = np.sum(row)
            norm_matrix[k, :] = row / s
        else:  # 'мах'
            inv_row = 1.0 / row
            s = np.sum(inv_row)
            norm_matrix[k, :] = inv_row / s

    return norm_matrix

# =============================================================================
# 3. Нелінійна схема компромісів Вороніна — інтегрована оцінка
# =============================================================================

def voronin_score(norm_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n_criteria, n_products = norm_matrix.shape
    scores = np.zeros(n_products, dtype=float)

    for j in range(n_products):
        for k in range(n_criteria):
            f = norm_matrix[k, j]
            # Захист від f >= 1 (виникає у крайніх значень)
            f = min(f, 0.9999)
            scores[j] += weights[k] * (1.0 - f) ** (-1)

    return scores

# =============================================================================
# 4. Головна функція
# =============================================================================

def evaluate(raw_weights: list[float]) -> None:
    # --- 1. Генерація даних ---
    matrix, criteria_names, product_names, criteria_type = generate_data()

    # --- 2. Нормалізація матриці ---
    norm_matrix = normalize_matrix(matrix, criteria_type)

    # --- 3. Нормалізація ваг ---
    w = np.array(raw_weights, dtype=float)
    weights = w / w.sum()

    # --- 4. Інтегровані оцінки ---
    scores = voronin_score(norm_matrix, weights)

    # --- 5. Результати ---
    print("\nНОРМАЛІЗОВАНА МАТРИЦЯ КРИТЕРІЇВ:")
    print("-" * 70)
    header = f"{'Критерій':<25} {'Тип':<5} {'Вага':>6}"
    print(header)
    print("-" * 70)
    for k, (name, ctype, wk) in enumerate(zip(criteria_names, criteria_type, weights)):
        print(f"  {name:<23} {ctype:<5} {wk:>6.4f}")

    print("\nОЦІНКА ЗА МЕТОДОМ ВОРОНІНА:")
    print("-" * 70)
    print(f"  {'Товар':<20} {'Оцінка':>12}  {'Ранг':>6}")
    print("-" * 70)

    ranked_indices = np.argsort(scores)
    ranks = np.empty_like(ranked_indices)
    ranks[ranked_indices] = np.arange(1, len(scores) + 1)

    for j, (name, score, rank) in enumerate(zip(product_names, scores, ranks)):
        marker = " <<< ОПТИМАЛЬНИЙ" if rank == 1 else ""
        print(f"  {name:<20} {score:>12.4f}  {rank:>6}{marker}")

    best_idx = int(np.argmin(scores))
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТ АНАЛІЗУ:")
    print("=" * 70)
    print(f"  Оптимальний товар : {product_names[best_idx]}")
    print(f"  Оцінка (мін = краще)    : {scores[best_idx]:.4f}")
    print("=" * 70)

    # Топ-5 найкращих
    print("\nТОП-5 НАЙКРАЩИХ:")
    print("-" * 50)
    for rank_pos, idx in enumerate(ranked_indices[:5], start=1):
        print(f"  {rank_pos}. {product_names[idx]:<20}  оцінка = {scores[idx]:.4f}")
    print("-" * 50)

    # --- 6. Побудова графіка ---
    plt.figure(figsize=(10, 6))
    
    # Сортуємо для графіка, найкращі (з найменшою оцінкою) зверху
    sorted_plot_indices = ranked_indices[::-1] # Зворотний порядок (найгірші знизу, найкращі зверху)
    
    sorted_names = [product_names[i] for i in sorted_plot_indices]
    sorted_scores = [scores[i] for i in sorted_plot_indices]

    bars = plt.barh(sorted_names, sorted_scores, color='salmon', edgecolor='black')
    
    for bar in bars:
        plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, 
                 f'{bar.get_width():.3f}', 
                 va='center', ha='left', fontsize=10)

    plt.xlabel('Інтегрована оцінка (за методом Вороніна, менше = краще)')
    plt.ylabel('Товари')
    plt.title('Інтегрована оцінка ефективності 10 товарів (метод Вороніна)')
    
    # Додаємо місце справа для підписів
    max_score = max(scores)
    plt.xlim(0, max_score * 1.15)
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_img = 'efficiency_scores_voronin.png'
    plt.savefig(output_img, dpi=300)
    print(f"\nГрафік збережено як '{output_img}'")


if __name__ == "__main__":
    WEIGHTS = [
        0.4,  # C1 Прибутковість (%)    (мах)
        0.4,  # C2 Обсяг продажів (шт)  (мах)
        0.2,  # C3 Відсоток браку (%)   (мін)
    ]

    evaluate(WEIGHTS)
