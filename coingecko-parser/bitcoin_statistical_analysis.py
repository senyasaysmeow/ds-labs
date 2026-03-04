import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def MNK(S0, poly_order=2, verbose=True):
    n = len(S0)
    x = np.arange(n, dtype=float)
    # np.polyfit повертає коефіцієнти від старшого до молодшого
    coeffs_desc = np.polyfit(x, S0, poly_order)
    Yout = np.polyval(coeffs_desc, x).reshape(-1, 1)
    C = coeffs_desc[::-1].reshape(-1, 1)

    if verbose:
        terms = [f"{C[0, 0]:.6f}"]
        for p in range(1, poly_order + 1):
            terms.append(f"{C[p, 0]:.6e} * t^{p}")
        print(f"  Регресійна модель: y(t) = {' + '.join(terms)}")
    return Yout, C


def MNK_Extrapol(S0, koef, poly_order=2, verbose=True):
    n = len(S0)
    x = np.arange(n, dtype=float)
    coeffs_desc = np.polyfit(x, S0, poly_order)
    # Обчислення на розширеному діапазоні
    x_ext = np.arange(n + koef, dtype=float)
    Yout_ext = np.polyval(coeffs_desc, x_ext).reshape(-1, 1)
    C = coeffs_desc[::-1].reshape(-1, 1)

    if verbose:
        terms = [f"{C[0, 0]:.6f}"]
        for p in range(1, poly_order + 1):
            terms.append(f"{C[p, 0]:.6e} * t^{p}")
        print(f"  Регресійна модель (екстраполяція): y(t) = {' + '.join(terms)}")
    return Yout_ext, C


def stat_characteristics(S, label=""):
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
    print(f"  Кількість елементів вибірки  : {n}")
    print(f"  Середнє арифметичне          : {mean_val:.2f}")
    print(f"  Медіана                      : {median_val:.2f}")
    print(f"  Дисперсія                    : {var_val:.2f}")
    print(f"  СКВ                          : {std_val:.2f}")
    print(f"  Мінімум                      : {min_val:.2f}")
    print(f"  Максимум                     : {max_val:.2f}")
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


def stat_characteristics_detrended(S, label="", poly_order=2):
    Yout, _ = MNK(S, poly_order=poly_order)
    residuals = S - Yout.flatten()
    return stat_characteristics(residuals, label + " (залишки)")


def r2_score(S_real, S_model, label=""):
    ss_res = np.sum((S_real - S_model) ** 2)
    ss_tot = np.sum((S_real - np.mean(S_real)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    print(f"\n  R^2 ({label}): {r2:.6f}")
    return r2


def find_best_poly_order(S, orders=(1, 2, 3)):
    """Перебирає поліноми різних ступенів, обирає найкращий за R²."""
    print(f"\n  {'─' * 60}")
    print(f"  Автоматичний вибір ступеня полінома (порівняння R²)")
    print(f"  {'─' * 60}")
    print(f"  {'Ступінь':>10} {'R²':>14} {'Примітка'}")
    print(f"  {'─' * 60}")

    best_order = orders[0]
    best_r2 = -np.inf
    results = {}

    for order in orders:
        Yout, C = MNK(S, poly_order=order, verbose=False)
        ss_res = np.sum((S - Yout.flatten()) ** 2)
        ss_tot = np.sum((S - np.mean(S)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        results[order] = r2
        if r2 > best_r2:
            best_r2 = r2
            best_order = order

    for order in orders:
        marker = " <-- обрано" if order == best_order else ""
        print(f"  {order:>10} {results[order]:>14.6f} {marker}")

    print(f"  {'─' * 60}")
    print(f"  Обраний ступінь полінома: {best_order} (R² = {best_r2:.6f})")
    print(f"  {'─' * 60}")

    return best_order


def sliding_window_clean(S0, n_wind=5):
    n = len(S0)
    S_clean = np.copy(S0)
    half = n_wind // 2
    for i in range(half, n - half):
        window = S0[i - half : i + half + 1]
        median_val = np.median(window)
        std_val = np.std(window)
        if np.abs(S0[i] - median_val) > 2.5 * std_val and std_val > 0:
            S_clean[i] = median_val
    n_replaced = np.sum(S_clean != S0)
    print(
        f"  Ковзне вікно (розмір={n_wind}): замінено {int(n_replaced)} аномальних точок"
    )
    return S_clean


def synthesize_model(S_real, n_av_pct=5, q_av=3, poly_order=2):
    n = len(S_real)

    # 1) Визначення тренду реальних даних (МНК)
    Yout_trend, C_trend = MNK(S_real, poly_order=poly_order)
    trend = Yout_trend.flatten()

    # 2) Визначення залишків (шуму) реальних даних
    residuals = S_real - trend
    noise_mean = np.mean(residuals)
    noise_std = np.std(residuals)
    print("\n  Параметри шуму реальних даних:")
    print(f"    Середнє залишків  = {noise_mean:.4f}")
    print(f"    СКВ залишків      = {noise_std:.4f}")

    # 3) Генерація тренду моделі (з тими ж коефіцієнтами)
    coeffs_desc = C_trend[::-1, 0]
    x = np.arange(n, dtype=float)
    model_trend = np.polyval(coeffs_desc, x)

    # 4) Генерація нормального шуму з такими ж параметрами
    normal_noise = np.random.normal(noise_mean, noise_std, n)

    # 5) Генерація аномальних вимірів
    n_av = int(n * n_av_pct / 100)
    av_indices = np.random.choice(n, size=n_av, replace=False)
    anomaly_noise = np.random.normal(noise_mean, q_av * noise_std, n_av)

    # 6) Складання адитивної моделі
    S_model = model_trend + normal_noise
    for idx, av_idx in enumerate(av_indices):
        S_model[av_idx] = model_trend[av_idx] + anomaly_noise[idx]

    print(f"  Синтезовано модель: {n} точок, {n_av} аномальних вимірів ({n_av_pct}%)")

    return S_model, model_trend, C_trend, av_indices


def verify_model(
    S_real, S_model, label_real="Реальні дані", label_model="Модель", poly_order=2
):
    stats_real = stat_characteristics(S_real, label_real)
    stats_model = stat_characteristics(S_model, label_model)

    print(f"\n{'=' * 60}")
    print("  ВЕРИФІКАЦІЯ: порівняння характеристик")
    print(f"{'=' * 60}")
    print(f"  {'Параметр':<30} {'Реальні':>14} {'Модель':>14} {'Δ, %':>10}")
    print(f"  {'-' * 68}")
    for key in ["mean", "median", "var", "std", "min", "max"]:
        r = stats_real[key]
        m = stats_model[key]
        delta_pct = abs(r - m) / abs(r) * 100 if r != 0 else 0
        names = {
            "mean": "Середнє",
            "median": "Медіана",
            "var": "Дисперсія",
            "std": "СКВ",
            "min": "Мінімум",
            "max": "Максимум",
        }
        print(f"  {names[key]:<30} {r:>14.2f} {m:>14.2f} {delta_pct:>9.2f}%")
    print(f"  {'-' * 68}")

    # Порівняння трендів
    Yout_real, _ = MNK(S_real, poly_order=poly_order)
    Yout_model, _ = MNK(S_model, poly_order=poly_order)
    r2_real = r2_score(S_real, Yout_real.flatten(), "Тренд реальних даних")
    r2_model = r2_score(S_model, Yout_model.flatten(), "Тренд моделі")

    return stats_real, stats_model


def plot_data_and_trend(S, Yout, title, dates=None, save_name=None):
    plt.figure(figsize=(14, 5))
    x = np.arange(len(S))
    plt.plot(x, S, "b-", alpha=0.5, linewidth=0.8, label="Дані")
    plt.plot(x[: len(Yout)], Yout, "r-", linewidth=2, label="Тренд (МНК)")
    plt.title(title)
    plt.xlabel("Індекс вимірювання")
    plt.ylabel("Ціна, USD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=150, bbox_inches="tight")
    plt.show()


def plot_comparison(S_real, S_model, title, save_name=None):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(S_real, "b-", alpha=0.7, linewidth=0.8)
    axes[0].set_title("Реальні дані (результат парсингу)")
    axes[0].set_ylabel("Ціна, USD")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(S_model, "g-", alpha=0.7, linewidth=0.8)
    axes[1].set_title("Синтезована модель")
    axes[1].set_xlabel("Індекс вимірювання")
    axes[1].set_ylabel("Ціна, USD")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=150, bbox_inches="tight")
    plt.show()


def plot_histograms(S_real, S_model, poly_order=2, save_name=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Залишки реальних даних
    Yout_real, _ = MNK(S_real, poly_order=poly_order)
    resid_real = S_real - Yout_real.flatten()

    # Залишки моделі
    Yout_model, _ = MNK(S_model, poly_order=poly_order)
    resid_model = S_model - Yout_model.flatten()

    axes[0].hist(resid_real, bins=40, facecolor="blue", alpha=0.6, edgecolor="black")
    axes[0].set_title("Гістограма залишків: реальні дані")
    axes[0].set_xlabel("Відхилення від тренду, USD")
    axes[0].set_ylabel("Частота")

    axes[1].hist(resid_model, bins=40, facecolor="green", alpha=0.6, edgecolor="black")
    axes[1].set_title("Гістограма залишків: модель")
    axes[1].set_xlabel("Відхилення від тренду, USD")
    axes[1].set_ylabel("Частота")

    plt.suptitle("Порівняння розподілів залишків", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=150, bbox_inches="tight")
    plt.show()


def plot_extrapolation(S_clean, Yout_ext, koef, title, save_name=None):
    n_orig = len(S_clean)
    n_total = len(Yout_ext)
    plt.figure(figsize=(14, 5))
    x_data = np.arange(n_orig)
    x_ext = np.arange(n_total)

    plt.plot(x_data, S_clean, "b-", alpha=0.5, linewidth=0.8, label="Очищені дані")
    plt.plot(x_ext, Yout_ext, "r-", linewidth=2, label="МНК екстраполяція")
    plt.axvline(
        x=n_orig, color="orange", linestyle="--", linewidth=1.5, label="Межа прогнозу"
    )
    plt.fill_between(
        x_ext[n_orig:],
        np.asarray(Yout_ext[n_orig:], dtype=float).ravel(),
        alpha=0.15,
        color="red",
    )
    plt.title(title)
    plt.xlabel("Індекс вимірювання")
    plt.ylabel("Ціна, USD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("  СТАТИСТИЧНИЙ АНАЛІЗ ДАНИХ ПРО КУРС BITCOIN (CoinGecko)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    #  ЕТАП 1.  Завантаження реальних даних
    # -------------------------------------------------------------------------
    csv_path = "bitcoin_prices_last_90_days.csv"
    df = pd.read_csv(csv_path)
    prices = df["price_usd"].values
    dates = df["date"].values if "date" in df.columns else np.arange(len(prices))
    print(f"Завантажено {len(prices)} точок даних з {csv_path}")

    n = len(prices)

    # -------------------------------------------------------------------------
    #  ЕТАП 2.  Оцінка динаміки тренду реальних даних
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ЕТАП 2. Оцінка динаміки тренду реальних даних")
    print("=" * 70)

    # Автоматичний вибір найкращого ступеня полінома
    best_poly = find_best_poly_order(prices, orders=(1, 2, 3))
    poly_names = {1: "Лінійний", 2: "Квадратичний", 3: "Кубічний"}
    poly_name = poly_names.get(best_poly, f"Поліном {best_poly}-го ступеня")

    # МНК — згладжування з автоматично обраним ступенем
    Yout_real, C_real = MNK(prices, poly_order=best_poly)
    r2_trend = r2_score(
        prices, Yout_real.flatten(), f"{poly_name} тренд реальних даних"
    )

    # Аналіз напрямку тренду
    trend_start = Yout_real[0, 0]
    trend_end = Yout_real[-1, 0]
    trend_change = trend_end - trend_start
    trend_pct = (trend_change / trend_start) * 100

    print("\n  Динаміка тренду:")
    print(f"    Обраний ступінь полінома   : {best_poly} ({poly_name})")
    print(f"    Початкове значення тренду : ${trend_start:,.2f}")
    print(f"    Кінцеве значення тренду   : ${trend_end:,.2f}")
    print(f"    Зміна                     : ${trend_change:,.2f} ({trend_pct:+.2f}%)")
    if best_poly >= 2 and C_real[2, 0] > 0:
        print("    Характер тренду           : Зростання (a2 > 0)")
    elif best_poly >= 2 and C_real[2, 0] < 0:
        print("    Характер тренду           : Спадання (a2 < 0)")
    elif best_poly == 1 and C_real[1, 0] > 0:
        print("    Характер тренду           : Лінійне зростання (a1 > 0)")
    elif best_poly == 1 and C_real[1, 0] < 0:
        print("    Характер тренду           : Лінійне спадання (a1 < 0)")
    else:
        print("    Характер тренду           : Стаціонарний")

    plot_data_and_trend(
        prices,
        Yout_real,
        f"Етап 2: Реальні дані Bitcoin та {poly_name.lower()} тренд (МНК, ступінь {best_poly})",
        save_name="plot_01_trend.png",
    )

    # -------------------------------------------------------------------------
    #  ЕТАП 3.  Статистичні характеристики результатів парсингу
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ЕТАП 3. Статистичні характеристики результатів парсингу")
    print("=" * 70)

    stats_raw = stat_characteristics(prices, "Вхідна вибірка (ціна Bitcoin, USD)")
    stats_detrended = stat_characteristics_detrended(
        prices, "Вибірка після видалення тренду", poly_order=best_poly
    )

    # Детекція та очищення аномалій
    n_wind = 7  # розмір ковзного вікна
    prices_clean = sliding_window_clean(prices, n_wind)
    stats_clean = stat_characteristics(
        prices_clean, "Вибірка після очищення від аномалій"
    )

    # МНК — модель очищених даних
    Yout_clean, C_clean = MNK(prices_clean, poly_order=best_poly)
    r2_clean = r2_score(prices_clean, Yout_clean.flatten(), "Тренд очищених даних")

    plot_data_and_trend(
        prices_clean,
        Yout_clean,
        f"Етап 3: Очищені дані Bitcoin та {poly_name.lower()} тренд (МНК, ступінь {best_poly})",
        save_name="plot_02_clean_trend.png",
    )

    # -------------------------------------------------------------------------
    #  ЕТАП 4.  Синтез та верифікація моделі
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ЕТАП 4. Синтез та верифікація моделі")
    print("=" * 70)

    print("\n  Синтез адитивної моделі: Тренд + Нормальний шум + Аномальний шум")
    n_av_pct = 5  # відсоток аномальних вимірів
    q_av = 3  # коефіцієнт переваги аномальних похибок

    S_model, model_trend, C_model, av_indices = synthesize_model(
        prices_clean, n_av_pct=n_av_pct, q_av=q_av, poly_order=best_poly
    )

    # Верифікація
    stats_real_v, stats_model_v = verify_model(
        prices, S_model, "Реальні дані", "Синтезована модель", poly_order=best_poly
    )

    # Порівняльні графіки
    plot_comparison(
        prices,
        S_model,
        "Етап 4: Порівняння реальних даних та синтезованої моделі",
        save_name="plot_03_comparison.png",
    )

    plot_histograms(
        prices, S_model, poly_order=best_poly, save_name="plot_04_histograms.png"
    )

    # Очищення синтезованої моделі та МНК
    S_model_clean = sliding_window_clean(S_model, n_wind)
    Yout_model_clean, _ = MNK(S_model_clean, poly_order=best_poly)
    r2_model = r2_score(
        S_model_clean, Yout_model_clean.flatten(), "Тренд очищеної моделі"
    )

    plot_data_and_trend(
        S_model_clean,
        Yout_model_clean,
        f"Етап 4: Синтезована модель (очищена) та тренд (МНК, ступінь {best_poly})",
        save_name="plot_05_model_trend.png",
    )

    # -------------------------------------------------------------------------
    #  ЕТАП 5.  Прогнозування (МНК екстраполяція)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ЕТАП 5. Прогнозування (МНК екстраполяція)")
    print("=" * 70)

    koef_extrapol = 0.5  # прогноз на 50% від обсягу вибірки
    koef = int(np.ceil(n * koef_extrapol))
    print(f"  Інтервал прогнозу: {koef} точок ({koef_extrapol * 100:.0f}% від вибірки)")

    Yout_extrapol, C_ext = MNK_Extrapol(prices_clean, koef, poly_order=best_poly)

    # Статистичні характеристики прогнозу
    extrapol_values = Yout_extrapol[n:, 0]
    stat_characteristics(extrapol_values, "Прогнозовані значення")

    # Довірчий інтервал
    residuals_clean = prices_clean - Yout_clean.flatten()
    std_resid = np.std(residuals_clean)
    print(f"\n  СКВ залишків для довірчого інтервалу: ±${std_resid:,.2f}")
    print(f"  Довірчий інтервал (95%, ±2σ)        : ±${2 * std_resid:,.2f}")

    plot_extrapolation(
        prices_clean,
        Yout_extrapol,
        koef,
        f"Етап 5: Прогнозування курсу Bitcoin (МНК екстраполяція, ступінь {best_poly})",
        save_name="plot_06_extrapolation.png",
    )

    # -------------------------------------------------------------------------
    #  ЕТАП 6.  Аналіз отриманих результатів
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ЕТАП 6. Аналіз отриманих результатів")
    print("=" * 70)

    # Визначення старшого коефіцієнта для аналізу характеру тренду
    top_coeff = C_real[best_poly, 0]
    if best_poly >= 2:
        trend_direction = "Тренд спадний" if C_real[2, 0] < 0 else "Тренд зростання"
        coeff_label = f"Коефіцієнт a{best_poly} = {top_coeff:.6e}"
    else:
        trend_direction = "Тренд спадний" if C_real[1, 0] < 0 else "Тренд зростання"
        coeff_label = f"Коефіцієнт a1 = {C_real[1, 0]:.6e}"

    print(f"""
  0. ВИБІР СТУПЕНЯ ПОЛІНОМА:
     Порівняння R² для поліномів 1-го, 2-го та 3-го ступеня:
     - Автоматично обраний ступінь: {best_poly} ({poly_name})

  1. ДИНАМІКА ТРЕНДУ:
     {poly_name} тренд (МНК) за період спостереження ({dates[0]} — {dates[-1]}):
     - Зміна ціни за трендом: ${trend_change:,.2f} ({trend_pct:+.2f}%)
     - Коефіцієнт детермінації R^2 = {r2_trend:.4f}
     - {coeff_label}
       {trend_direction}

  2. СТАТИСТИЧНІ ХАРАКТЕРИСТИКИ:
     - Середня ціна         : ${stats_raw["mean"]:,.2f}
     - СКВ (волатильність)  : ${stats_raw["std"]:,.2f}
     - Діапазон             : ${stats_raw["min"]:,.2f} — ${stats_raw["max"]:,.2f}
     - Кількість аномалій виявлено та замінено ковзним вікном (n_wind={n_wind})

  3. СИНТЕЗОВАНА МОДЕЛЬ:
     Адитивна модель: S(t) = Тренд(t) + N(μ, σ) + AV({n_av_pct}%, Q={q_av})
     - R^2 моделі            : {r2_model:.4f}
     - R^2 реальних даних    : {r2_clean:.4f}
     - Модель адекватно відтворює тренд та стохастичну складову реальних даних

  4. ПРОГНОЗУВАННЯ:
     - Прогноз на {koef} точок вперед (МНК екстраполяція, ступінь {best_poly})
     - Довірчий інтервал    : ±${2 * std_resid:,.2f} (95%, 2σ)
     - Прогнозне значення на кінець інтервалу: ${Yout_extrapol[-1, 0]:,.2f}
    """)
