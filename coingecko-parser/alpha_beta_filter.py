import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#  Утиліти
# ─────────────────────────────────────────────────────────────────────────────


def stat_characteristics(S, label=""):
    """Друкує та повертає статистичні характеристики вибірки."""
    n = len(S)
    mean_val = np.mean(S)
    median_val = np.median(S)
    var_val = np.var(S)
    std_val = np.std(S)
    min_val = np.min(S)
    max_val = np.max(S)
    print(f"\n{'=' * 60}")
    print(f"  Статистичні характеристики: {label}")
    print(f"{'=' * 60}")
    print(f"  Кількість елементів  : {n}")
    print(f"  Середнє              : {mean_val:.2f}")
    print(f"  Медіана              : {median_val:.2f}")
    print(f"  Дисперсія            : {var_val:.2f}")
    print(f"  СКВ                  : {std_val:.2f}")
    print(f"  Мінімум              : {min_val:.2f}")
    print(f"  Максимум             : {max_val:.2f}")
    print(f"{'=' * 60}")
    return {
        "n": n,
        "mean": mean_val,
        "median": median_val,
        "var": var_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
    }


def r2_score(S_real, S_filtered, label=""):
    """Коефіцієнт детермінації R²."""
    ss_res = np.sum((S_real - S_filtered) ** 2)
    ss_tot = np.sum((S_real - np.mean(S_real)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    print(f"  R² ({label}): {r2:.6f}")
    return r2


def mse_score(S_real, S_filtered, label=""):
    """Середньоквадратична похибка."""
    mse = np.mean((S_real - S_filtered) ** 2)
    print(f"  MSE ({label}): {mse:.4f}")
    return mse


# ─────────────────────────────────────────────────────────────────────────────
#  Фільтр альфа-бета
# ─────────────────────────────────────────────────────────────────────────────


def alpha_beta_filter(
    z,
    alpha,
    beta,
    dt=1.0,
    adaptive=True,
    k_sigma=3.0,
    innov_window=50,
    max_velocity=None,
):
    n = len(z)
    x_est = np.zeros(n)
    v_est = np.zeros(n)
    innovations = np.zeros(n)

    # Ініціалізація
    x_est[0] = z[0]
    v_est[0] = 0.0

    # Буфер нев'язок для адаптації
    innov_buf = []

    for k in range(1, n):
        # ── Прогноз ──
        x_pred = x_est[k - 1] + v_est[k - 1] * dt
        v_pred = v_est[k - 1]

        # ── Нев'язка (innovation) ──
        innov = z[k] - x_pred
        innovations[k] = innov

        # ── Адаптивне масштабування (захист від розбіжності) ──
        a_k = alpha
        b_k = beta

        if adaptive and len(innov_buf) >= innov_window:
            sigma_innov = np.std(innov_buf[-innov_window:])
            if sigma_innov > 0 and np.abs(innov) > k_sigma * sigma_innov:
                # Різке відхилення — тимчасово «відкриваємо» фільтр
                scale = min(np.abs(innov) / (k_sigma * sigma_innov), 1.0 / alpha)
                a_k = min(alpha * scale, 0.95)
                b_k = min(beta * scale, 0.95)

        innov_buf.append(innov)

        # ── Корекція ──
        x_est[k] = x_pred + a_k * innov
        v_est[k] = v_pred + (b_k / dt) * innov

        # ── Обмеження швидкості ──
        if max_velocity is not None:
            v_est[k] = np.clip(v_est[k], -max_velocity, max_velocity)

    return x_est, v_est, innovations


# ─────────────────────────────────────────────────────────────────────────────
#  Фільтр альфа-бета-гамма
# ─────────────────────────────────────────────────────────────────────────────


def alpha_beta_gamma_filter(
    z,
    alpha,
    beta,
    gamma,
    dt=1.0,
    adaptive=True,
    k_sigma=3.0,
    innov_window=50,
    max_velocity=None,
    max_acceleration=None,
):
    n = len(z)
    x_est = np.zeros(n)
    v_est = np.zeros(n)
    a_est = np.zeros(n)
    innovations = np.zeros(n)

    # Ініціалізація
    x_est[0] = z[0]
    v_est[0] = 0.0
    a_est[0] = 0.0

    innov_buf = []

    for k in range(1, n):
        # ── Прогноз ──
        x_pred = x_est[k - 1] + v_est[k - 1] * dt + 0.5 * a_est[k - 1] * dt**2
        v_pred = v_est[k - 1] + a_est[k - 1] * dt
        a_pred = a_est[k - 1]

        # ── Нев'язка ──
        innov = z[k] - x_pred
        innovations[k] = innov

        # ── Адаптивне масштабування ──
        a_k = alpha
        b_k = beta
        g_k = gamma

        if adaptive and len(innov_buf) >= innov_window:
            sigma_innov = np.std(innov_buf[-innov_window:])
            if sigma_innov > 0 and np.abs(innov) > k_sigma * sigma_innov:
                scale = min(np.abs(innov) / (k_sigma * sigma_innov), 1.0 / alpha)
                a_k = min(alpha * scale, 0.95)
                b_k = min(beta * scale, 0.95)
                g_k = min(gamma * scale, 0.95)

        innov_buf.append(innov)

        # ── Корекція ──
        x_est[k] = x_pred + a_k * innov
        v_est[k] = v_pred + (b_k / dt) * innov
        a_est[k] = a_pred + (2.0 * g_k / dt**2) * innov

        # ── Обмеження ──
        if max_velocity is not None:
            v_est[k] = np.clip(v_est[k], -max_velocity, max_velocity)
        if max_acceleration is not None:
            a_est[k] = np.clip(a_est[k], -max_acceleration, max_acceleration)

    return x_est, v_est, a_est, innovations


# ─────────────────────────────────────────────────────────────────────────────
#  Оптимальні коефіцієнти (критерій Бенедикта — Борднера)
# ─────────────────────────────────────────────────────────────────────────────


def optimal_ab_coefficients(alpha):
    """
    Оптимальне beta для альфа-бета фільтра за критерієм Бенедикта — Борднера:
        beta = alpha^2 / (2 - alpha)
    Забезпечує критичне демпфування (без коливань).
    """
    beta = alpha**2 / (2 - alpha)
    return alpha, beta


def optimal_abg_coefficients(alpha):
    """
    Оптимальні beta, gamma для альфа-бета-гамма фільтра
    (критерій мінімуму дисперсії при заданому alpha):
        beta  = alpha^2 / (2 - alpha)      (аналогічно AB)
        gamma = beta^2 / (2 * alpha)
    """
    beta = alpha**2 / (2 - alpha)
    gamma = beta**2 / (2 * alpha)
    return alpha, beta, gamma


# ─────────────────────────────────────────────────────────────────────────────
#  Побудова графіків
# ─────────────────────────────────────────────────────────────────────────────


def plot_filter_result(
    z,
    x_est,
    title,
    label_raw="Вхідний сигнал",
    label_filt="Згладжений сигнал",
    save_name=None,
):
    """Графік: вхідний сигнал vs. результат фільтрації."""
    plt.figure(figsize=(14, 5))
    x = np.arange(len(z))
    plt.plot(x, z, "b-", alpha=0.4, linewidth=0.7, label=label_raw)
    plt.plot(x, x_est, "r-", linewidth=1.5, label=label_filt)
    plt.title(title)
    plt.xlabel("Індекс вимірювання")
    plt.ylabel("Ціна, USD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=150, bbox_inches="tight")
    plt.show()


def plot_innovations(innovations, title, save_name=None):
    """Графік нев'язок (innovation sequence)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    x = np.arange(len(innovations))

    # Часовий ряд нев'язок
    axes[0].plot(x, innovations, "g-", alpha=0.7, linewidth=0.6)
    axes[0].axhline(y=0, color="k", linewidth=0.5)
    sigma = np.std(innovations)
    axes[0].axhline(
        y=3 * sigma,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"+3σ = {3 * sigma:.2f}",
    )
    axes[0].axhline(
        y=-3 * sigma,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"-3σ = {-3 * sigma:.2f}",
    )
    axes[0].set_title("Часовий ряд нев'язок (innovations)")
    axes[0].set_ylabel("Нев'язка, USD")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Гістограма нев'язок
    axes[1].hist(
        innovations,
        bins=60,
        facecolor="green",
        alpha=0.6,
        edgecolor="black",
        density=True,
    )
    # Теоретичний нормальний розподіл
    x_norm = np.linspace(innovations.min(), innovations.max(), 200)
    mu = np.mean(innovations)
    pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x_norm - mu) / sigma) ** 2
    )
    axes[1].plot(x_norm, pdf, "r-", linewidth=2, label=f"N({mu:.2f}, {sigma:.2f})")
    axes[1].set_title("Гістограма нев'язок")
    axes[1].set_xlabel("Нев'язка, USD")
    axes[1].set_ylabel("Щільність")
    axes[1].legend()

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=150, bbox_inches="tight")
    plt.show()


def plot_velocity(v_est, title, save_name=None):
    """Графік оцінки швидкості."""
    plt.figure(figsize=(14, 4))
    plt.plot(v_est, "m-", linewidth=0.8, alpha=0.8)
    plt.axhline(y=0, color="k", linewidth=0.5)
    plt.title(title)
    plt.xlabel("Індекс вимірювання")
    plt.ylabel("Швидкість (USD/крок)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=150, bbox_inches="tight")
    plt.show()


def plot_acceleration(a_est, title, save_name=None):
    """Графік оцінки прискорення."""
    plt.figure(figsize=(14, 4))
    plt.plot(a_est, "c-", linewidth=0.8, alpha=0.8)
    plt.axhline(y=0, color="k", linewidth=0.5)
    plt.title(title)
    plt.xlabel("Індекс вимірювання")
    plt.ylabel("Прискорення (USD/крок²)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=150, bbox_inches="tight")
    plt.show()


def plot_comparison_ab_abg(z, x_ab, x_abg, title, save_name=None):
    """Порівняння результатів двох фільтрів на одному графіку."""
    plt.figure(figsize=(14, 6))
    x = np.arange(len(z))
    plt.plot(x, z, "b-", alpha=0.3, linewidth=0.6, label="Вхідний сигнал")
    plt.plot(x, x_ab, "r-", linewidth=1.3, label="α-β фільтр")
    plt.plot(x, x_abg, "g-", linewidth=1.3, label="α-β-γ фільтр")
    plt.title(title)
    plt.xlabel("Індекс вимірювання")
    plt.ylabel("Ціна, USD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=150, bbox_inches="tight")
    plt.show()


def plot_divergence_demo(z, x_no_protect, x_protected, title, save_name=None):
    """Демонстрація ефекту захисту від розбіжності."""
    plt.figure(figsize=(14, 6))
    x = np.arange(len(z))
    plt.plot(x, z, "b-", alpha=0.3, linewidth=0.6, label="Вхідний сигнал")
    plt.plot(
        x,
        x_no_protect,
        "r--",
        linewidth=1.2,
        alpha=0.8,
        label="Без захисту від розбіжності",
    )
    plt.plot(x, x_protected, "g-", linewidth=1.5, label="З адаптивним захистом")
    plt.title(title)
    plt.xlabel("Індекс вимірювання")
    plt.ylabel("Ціна, USD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Головний блок
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  РЕКУРЕНТНЕ ЗГЛАДЖУВАННЯ α-β / α-β-γ ФІЛЬТРОМ")
    print("  (з заходами подолання розбіжності)")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────
    #  ЕТАП 1. Завантаження даних
    # ─────────────────────────────────────────────────────────────────────
    csv_path = "bitcoin_prices_last_90_days.csv"
    df = pd.read_csv(csv_path)
    prices = df["price_usd"].values.astype(float)
    n = len(prices)
    print(f"\n  Завантажено {n} точок даних з {csv_path}")

    # ─────────────────────────────────────────────────────────────────────
    #  ЕТАП 2. Вибір моделі
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ЕТАП 2. Вибір моделі фільтрації")
    print("=" * 70)
    print("""
  Доступні моделі:
    1 — α-β фільтр    (модель з постійною швидкістю)
    2 — α-β-γ фільтр  (модель з постійним прискоренням)
    3 — обидві моделі  (порівняльний аналіз)
    """)

    while True:
        choice = input("  Оберіть модель (1 / 2 / 3): ").strip()
        if choice in ("1", "2", "3"):
            break
        print("  Невірний вибір. Спробуйте ще.")

    model_choice = int(choice)

    # ─────────────────────────────────────────────────────────────────────
    #  ЕТАП 3. Налаштування параметрів
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ЕТАП 3. Налаштування параметрів фільтра")
    print("=" * 70)

    # Параметр згладжування alpha (0..1): менше → більше згладжування
    alpha = 0.15
    dt = 1.0  # крок дискретизації (1 вимірювання)

    # Обмеження для захисту від розбіжності
    price_range = np.max(prices) - np.min(prices)
    max_velocity = price_range * 0.05  # макс. зміна за крок — 5% діапазону
    max_acceleration = price_range * 0.01  # макс. прискорення — 1% діапазону

    # Оптимальні коефіцієнти за Бенедиктом — Борднером
    alpha_ab, beta_ab = optimal_ab_coefficients(alpha)
    alpha_abg, beta_abg, gamma_abg = optimal_abg_coefficients(alpha)

    print(f"\n  Базовий параметр alpha = {alpha}")
    print(f"  Крок дискретизації dt  = {dt}")
    print(f"\n  α-β фільтр (Бенедикт — Борднер):")
    print(f"    alpha = {alpha_ab:.6f}")
    print(f"    beta  = {beta_ab:.6f}")
    print(f"\n  α-β-γ фільтр (оптимальний):")
    print(f"    alpha = {alpha_abg:.6f}")
    print(f"    beta  = {beta_abg:.6f}")
    print(f"    gamma = {gamma_abg:.6f}")
    print(f"\n  Захист від розбіжності:")
    print(f"    Макс. швидкість     = {max_velocity:.2f} USD/крок")
    print(f"    Макс. прискорення   = {max_acceleration:.2f} USD/крок²")
    print(f"    Адаптивний поріг    = 3σ нев'язки (ковзне вікно = 50)")

    # ─────────────────────────────────────────────────────────────────────
    #  ЕТАП 4. Фільтрація
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ЕТАП 4. Рекурентне згладжування")
    print("=" * 70)

    results = {}

    # ── α-β фільтр ──
    if model_choice in (1, 3):
        print("\n  ── Альфа-бета фільтр ──")

        # З адаптивним захистом від розбіжності
        x_ab, v_ab, innov_ab = alpha_beta_filter(
            prices,
            alpha_ab,
            beta_ab,
            dt=dt,
            adaptive=True,
            k_sigma=3.0,
            innov_window=50,
            max_velocity=max_velocity,
        )

        # Без захисту (для демонстрації)
        x_ab_noprot, _, _ = alpha_beta_filter(
            prices,
            alpha_ab,
            beta_ab,
            dt=dt,
            adaptive=False,
            max_velocity=None,
        )

        results["ab"] = {
            "x_est": x_ab,
            "v_est": v_ab,
            "innovations": innov_ab,
            "x_noprot": x_ab_noprot,
        }

        stat_characteristics(prices, "Вхідний сигнал (вимірювання)")
        stat_characteristics(x_ab, "α-β фільтр (згладжені)")

        mse_ab = mse_score(prices, x_ab, "α-β фільтр")
        r2_ab = r2_score(prices, x_ab, "α-β фільтр")

        # Графіки
        plot_filter_result(
            prices,
            x_ab,
            f"α-β фільтр: згладжування (α={alpha_ab:.4f}, β={beta_ab:.4f})",
            save_name="plot_ab_filter.png",
        )
        plot_innovations(
            innov_ab,
            "α-β фільтр: аналіз нев'язок (innovations)",
            save_name="plot_ab_innovations.png",
        )
        plot_velocity(
            v_ab,
            "α-β фільтр: оцінка швидкості зміни ціни",
            save_name="plot_ab_velocity.png",
        )
        plot_divergence_demo(
            prices,
            x_ab_noprot,
            x_ab,
            "α-β фільтр: порівняння з/без захисту від розбіжності",
            save_name="plot_ab_divergence.png",
        )

    # ── α-β-γ фільтр ──
    if model_choice in (2, 3):
        print("\n  ── Альфа-бета-гамма фільтр ──")

        # З адаптивним захистом від розбіжності
        x_abg, v_abg, a_abg, innov_abg = alpha_beta_gamma_filter(
            prices,
            alpha_abg,
            beta_abg,
            gamma_abg,
            dt=dt,
            adaptive=True,
            k_sigma=3.0,
            innov_window=50,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
        )

        # Без захисту
        x_abg_noprot, _, _, _ = alpha_beta_gamma_filter(
            prices,
            alpha_abg,
            beta_abg,
            gamma_abg,
            dt=dt,
            adaptive=False,
            max_velocity=None,
            max_acceleration=None,
        )

        results["abg"] = {
            "x_est": x_abg,
            "v_est": v_abg,
            "a_est": a_abg,
            "innovations": innov_abg,
            "x_noprot": x_abg_noprot,
        }

        stat_characteristics(prices, "Вхідний сигнал (вимірювання)")
        stat_characteristics(x_abg, "α-β-γ фільтр (згладжені)")

        mse_abg = mse_score(prices, x_abg, "α-β-γ фільтр")
        r2_abg = r2_score(prices, x_abg, "α-β-γ фільтр")

        # Графіки
        plot_filter_result(
            prices,
            x_abg,
            f"α-β-γ фільтр: згладжування (α={alpha_abg:.4f}, β={beta_abg:.4f}, γ={gamma_abg:.6f})",
            save_name="plot_abg_filter.png",
        )
        plot_innovations(
            innov_abg,
            "α-β-γ фільтр: аналіз нев'язок (innovations)",
            save_name="plot_abg_innovations.png",
        )
        plot_velocity(
            v_abg,
            "α-β-γ фільтр: оцінка швидкості зміни ціни",
            save_name="plot_abg_velocity.png",
        )
        plot_acceleration(
            a_abg,
            "α-β-γ фільтр: оцінка прискорення зміни ціни",
            save_name="plot_abg_acceleration.png",
        )
        plot_divergence_demo(
            prices,
            x_abg_noprot,
            x_abg,
            "α-β-γ фільтр: порівняння з/без захисту від розбіжності",
            save_name="plot_abg_divergence.png",
        )

    # ─────────────────────────────────────────────────────────────────────
    #  ЕТАП 5. Порівняльний аналіз (якщо обрано обидві моделі)
    # ─────────────────────────────────────────────────────────────────────
    if model_choice == 3:
        print("\n" + "=" * 70)
        print("  ЕТАП 5. Порівняльний аналіз α-β та α-β-γ фільтрів")
        print("=" * 70)

        plot_comparison_ab_abg(
            prices,
            results["ab"]["x_est"],
            results["abg"]["x_est"],
            "Порівняння: α-β vs. α-β-γ фільтри",
            save_name="plot_comparison_ab_abg.png",
        )

        # Таблиця порівняння
        mse_ab = np.mean((prices - results["ab"]["x_est"]) ** 2)
        mse_abg = np.mean((prices - results["abg"]["x_est"]) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)

        r2_ab = 1 - np.sum((prices - results["ab"]["x_est"]) ** 2) / ss_tot
        r2_abg = 1 - np.sum((prices - results["abg"]["x_est"]) ** 2) / ss_tot

        std_innov_ab = np.std(results["ab"]["innovations"])
        std_innov_abg = np.std(results["abg"]["innovations"])

        print(f"\n  {'Метрика':<40} {'α-β':>14} {'α-β-γ':>14}")
        print(f"  {'─' * 68}")
        print(
            f"  {'MSE (середньоквадратична похибка)':<40} {mse_ab:>14.4f} {mse_abg:>14.4f}"
        )
        print(f"  {'R² (коефіцієнт детермінації)':<40} {r2_ab:>14.6f} {r2_abg:>14.6f}")
        skv_label = "СКВ нев'язок"
        print(f"  {skv_label:<40} {std_innov_ab:>14.4f} {std_innov_abg:>14.4f}")
        print(
            f"  {'Параметри фільтра':<40} {'α=' + f'{alpha_ab:.4f}':>14} {'α=' + f'{alpha_abg:.4f}':>14}"
        )
        print(f"  {'':40} {'β=' + f'{beta_ab:.4f}':>14} {'β=' + f'{beta_abg:.4f}':>14}")
        print(f"  {'':40} {'':>14} {'γ=' + f'{gamma_abg:.6f}':>14}")
        print(f"  {'─' * 68}")

        # Визначення кращого фільтра
        if mse_ab < mse_abg:
            print(
                "\n  Висновок: α-β фільтр забезпечує менше MSE для даного набору даних."
            )
            print("  Це пояснюється тим, що модель постійної швидкості достатня")
            print("  для згладжування ціни Bitcoin з погодинною дискретизацією.")
        else:
            print(
                "\n  Висновок: α-β-γ фільтр забезпечує менше MSE для даного набору даних."
            )
            print("  Модель з прискоренням краще відстежує нелінійну динаміку ціни.")

    # ─────────────────────────────────────────────────────────────────────
    #  ЕТАП 6. Підсумки
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ПІДСУМКИ")
    print("=" * 70)

    print(f"""
  1. ЗАХОДИ ПОДОЛАННЯ РОЗБІЖНОСТІ ФІЛЬТРА:
     a) Оптимальні коефіцієнти за критерієм Бенедикта — Борднера:
        - Забезпечують критичне демпфування (відсутність коливань)
        - beta = alpha² / (2 - alpha)
        - gamma = beta² / (2 * alpha)

     b) Адаптивне масштабування коефіцієнтів:
        - Контроль нев'язки (innovation) на кожному кроці
        - Якщо |нев'язка| > {3.0}σ — тимчасове збільшення коефіцієнтів
        - Ковзне вікно ({50} точок) для оцінки поточної σ нев'язки
        - Обмеження масштабованих коефіцієнтів: max = 0.95

     c) Обмеження оцінок стану:
        - Максимальна швидкість:    {max_velocity:.2f} USD/крок (5% діапазону)
        - Максимальне прискорення:  {max_acceleration:.2f} USD/крок² (1% діапазону)
        - Запобігає необмеженому зростанню внутрішніх змінних фільтра

  2. РЕЗУЛЬТАТИ ФІЛЬТРАЦІЇ:""")

    if "ab" in results:
        mse_v = np.mean((prices - results["ab"]["x_est"]) ** 2)
        print(f"     α-β фільтр:   MSE = {mse_v:.4f}")

    if "abg" in results:
        mse_v = np.mean((prices - results["abg"]["x_est"]) ** 2)
        print(f"     α-β-γ фільтр: MSE = {mse_v:.4f}")
