import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def prepare_features(df, window_size=24):
    df_features = df.copy()

    # Ковзна волатильність (СКВ за вікном)
    df_features["volatility"] = (
        df_features["price_usd"].rolling(window=window_size).std()
    )

    # Відносна зміна ціни (%)
    df_features["price_change"] = df_features["price_usd"].pct_change() * 100

    # Відносна зміна обсягу (%)
    df_features["volume_change"] = df_features["volume_usd"].pct_change() * 100

    # Ковзне середнє ціни
    df_features["price_ma"] = (
        df_features["price_usd"].rolling(window=window_size).mean()
    )

    # Відхилення від ковзного середнього
    df_features["price_deviation"] = (
        (df_features["price_usd"] - df_features["price_ma"])
        / df_features["price_ma"]
        * 100
    )

    # Видаляємо NaN (перші window_size точок)
    df_features = df_features.dropna()

    print(f"  Підготовлено {len(df_features)} точок даних для кластеризації")
    print(f"  Видалено {len(df) - len(df_features)} точок через ковзне вікно")

    return df_features


def kmeans_clustering(df_features, n_clusters, features_list=None):
    print(f"\n{'=' * 60}")
    print("  K-MEANS КЛАСТЕРИЗАЦІЯ")
    print(f"{'=' * 60}")
    print(f"  Ознаки: {', '.join(features_list)}")
    print(f"  Кількість кластерів: {n_clusters}")

    # Вибираємо ознаки
    X = df_features[features_list].values

    # Нормалізація даних
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Пошук оптимальної кількості кластерів (метод ліктя)
    print("\n  Пошук оптимальної кількості кластерів (метод ліктя):")
    inertias = []
    silhouettes = []
    K_range = range(2, 8)

    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_temp = kmeans_temp.fit_predict(X_scaled)
        inertias.append(kmeans_temp.inertia_)
        sil = silhouette_score(X_scaled, labels_temp)
        silhouettes.append(sil)
        print(f"    K={k}: Inertia={kmeans_temp.inertia_:.0f}, Silhouette={sil:.4f}")

    # Обираємо K з найкращим силуетним коефіцієнтом
    best_k_idx = np.argmax(silhouettes)
    best_k = list(K_range)[best_k_idx]
    print(f"\n  Рекомендована кількість кластерів за Silhouette: {best_k}")

    # Фінальна кластеризація
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Додаємо мітки до датафрейму
    df_features = df_features.copy()
    df_features["kmeans_cluster"] = labels

    # Аналіз кластерів
    print("\n  Характеристики кластерів:")
    print(f"  {'-' * 70}")
    print(
        f"  {'Кластер':<10} {'Точок':<10} {'Середня ціна':<15} {'Середній обсяг':<20} {'Волатильність':<15}"
    )
    print(f"  {'-' * 70}")

    cluster_names = {}
    for cluster_id in range(n_clusters):
        cluster_data = df_features[df_features["kmeans_cluster"] == cluster_id]
        count = len(cluster_data)
        mean_price = cluster_data["price_usd"].mean()
        mean_volume = cluster_data["volume_usd"].mean()
        mean_volatility = cluster_data["volatility"].mean()
        mean_change = cluster_data["price_change"].mean()

        # Визначення типу кластера
        if mean_change > 0.1 and mean_volatility < df_features["volatility"].median():
            cluster_type = "Бичачий"
        elif (
            mean_change < -0.1 and mean_volatility < df_features["volatility"].median()
        ):
            cluster_type = "Ведмежий"
        elif mean_volatility > df_features["volatility"].quantile(0.75):
            cluster_type = "Волатильний"
        else:
            cluster_type = "Консолідація"

        cluster_names[cluster_id] = cluster_type
        print(
            f"  {cluster_id:<10} {count:<10} ${mean_price:>12,.0f} ${mean_volume:>17,.0f} ${mean_volatility:>12,.0f}"
        )

    print(f"  {'-' * 70}")

    # Інтерпретація кластерів
    print("\n  Інтерпретація кластерів:")
    for cluster_id, cluster_type in cluster_names.items():
        cluster_data = df_features[df_features["kmeans_cluster"] == cluster_id]
        print(
            f"    Кластер {cluster_id} ({cluster_type}): {len(cluster_data)} точок ({len(cluster_data) / len(df_features) * 100:.1f}%)"
        )

    # Якість кластеризації
    silhouette = silhouette_score(X_scaled, labels)
    print("\n  Якість кластеризації:")
    print(f"    Silhouette Score: {silhouette:.4f}")
    print("    (Значення близьке до 1 = добре розділені кластери)")

    return df_features, kmeans, scaler, inertias, silhouettes, list(K_range)


def dbscan_clustering(df_features, eps=0.5, min_samples=10, features_list=None):
    if features_list is None:
        features_list = ["price_change", "volume_change", "price_deviation"]

    print(f"\n{'=' * 60}")
    print("  DBSCAN КЛАСТЕРИЗАЦІЯ (ВИЯВЛЕННЯ АНОМАЛІЙ)")
    print(f"{'=' * 60}")
    print(f"  Ознаки: {', '.join(features_list)}")
    print(f"  Параметри: eps={eps}, min_samples={min_samples}")

    # Вибираємо ознаки
    X = df_features[features_list].values

    # Нормалізація даних
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # DBSCAN кластеризація
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    # Додаємо мітки до датафрейму
    df_features = df_features.copy()
    df_features["dbscan_cluster"] = labels

    # Аналіз результатів
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_anomalies = (labels == -1).sum()

    print("\n  Результати DBSCAN:")
    print(f"    Виявлено кластерів: {n_clusters}")
    print(
        f"    Виявлено аномалій (шум): {n_anomalies} ({n_anomalies / len(labels) * 100:.2f}%)"
    )

    # Характеристики аномалій
    if n_anomalies > 0:
        anomalies = df_features[df_features["dbscan_cluster"] == -1]
        print("\n  Характеристики аномальних точок:")
        print(f"    Середня зміна ціни     : {anomalies['price_change'].mean():+.2f}%")
        print(
            f"    Мін/Макс зміна ціни    : {anomalies['price_change'].min():+.2f}% / {anomalies['price_change'].max():+.2f}%"
        )
        print(f"    Середня зміна обсягу   : {anomalies['volume_change'].mean():+.2f}%")
        print(
            f"    Середнє відхилення     : {anomalies['price_deviation'].mean():+.2f}%"
        )

        # Топ-5 найбільших аномалій за зміною ціни
        print("\n  Топ-5 аномалій за зміною ціни:")
        top_anomalies = anomalies.nlargest(5, "price_change", keep="first")
        for _, row in top_anomalies.iterrows():
            date_str = row.get("datetime", row.get("date", "N/A"))
            print(
                f"    {date_str}: ціна ${row['price_usd']:,.0f}, зміна {row['price_change']:+.2f}%"
            )

        # Топ-5 найбільших падінь
        print("\n  Топ-5 аномалій за падінням ціни:")
        bottom_anomalies = anomalies.nsmallest(5, "price_change", keep="first")
        for _, row in bottom_anomalies.iterrows():
            date_str = row.get("datetime", row.get("date", "N/A"))
            print(
                f"    {date_str}: ціна ${row['price_usd']:,.0f}, зміна {row['price_change']:+.2f}%"
            )

    return df_features, dbscan, scaler


def plot_kmeans_results(
    df_features, inertias, silhouettes, K_range, save_prefix="plot_clustering"
):
    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Метод ліктя (Inertia)
    axes[0, 0].plot(K_range, inertias, "bo-", linewidth=2, markersize=8)
    axes[0, 0].set_xlabel("Кількість кластерів K")
    axes[0, 0].set_ylabel("Inertia (сума квадратів відстаней)")
    axes[0, 0].set_title("Метод ліктя для вибору K")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Silhouette Score
    axes[0, 1].plot(K_range, silhouettes, "go-", linewidth=2, markersize=8)
    axes[0, 1].set_xlabel("Кількість кластерів K")
    axes[0, 1].set_ylabel("Silhouette Score")
    axes[0, 1].set_title("Силуетний аналіз для вибору K")
    axes[0, 1].grid(True, alpha=0.3)
    best_k = K_range[np.argmax(silhouettes)]
    axes[0, 1].axvline(
        x=best_k, color="red", linestyle="--", label=f"Найкраще K={best_k}"
    )
    axes[0, 1].legend()

    # 3. Кластери на графіку ціни
    colors = plt.cm.viridis(np.linspace(0, 1, df_features["kmeans_cluster"].nunique()))
    for cluster_id in sorted(df_features["kmeans_cluster"].unique()):
        cluster_data = df_features[df_features["kmeans_cluster"] == cluster_id]
        axes[1, 0].scatter(
            range(len(cluster_data)),
            cluster_data["price_usd"],
            c=[colors[cluster_id]],
            label=f"Кластер {cluster_id}",
            alpha=0.6,
            s=10,
        )
    axes[1, 0].set_xlabel("Індекс точки")
    axes[1, 0].set_ylabel("Ціна, USD")
    axes[1, 0].set_title("Розподіл кластерів за ціною")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Часовий ряд з кольорами кластерів
    x = np.arange(len(df_features))
    scatter = axes[1, 1].scatter(
        x,
        df_features["price_usd"],
        c=df_features["kmeans_cluster"],
        cmap="viridis",
        alpha=0.6,
        s=10,
    )
    axes[1, 1].set_xlabel("Індекс вимірювання")
    axes[1, 1].set_ylabel("Ціна, USD")
    axes[1, 1].set_title("Часовий ряд Bitcoin з кластерами")
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label="Кластер")

    plt.suptitle("K-Means кластеризація даних Bitcoin", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_kmeans.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_dbscan_results(df_features, save_prefix="plot_clustering"):
    _, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Розділяємо нормальні точки та аномалії
    normal = df_features[df_features["dbscan_cluster"] != -1]
    anomalies = df_features[df_features["dbscan_cluster"] == -1]

    # 1. Ціна з аномаліями
    x_normal = np.arange(len(df_features))[df_features["dbscan_cluster"] != -1]
    x_anomaly = np.arange(len(df_features))[df_features["dbscan_cluster"] == -1]

    axes[0, 0].scatter(
        x_normal, normal["price_usd"], c="blue", alpha=0.3, s=5, label="Нормальні"
    )
    axes[0, 0].scatter(
        x_anomaly,
        anomalies["price_usd"],
        c="red",
        alpha=0.8,
        s=30,
        marker="x",
        label="Аномалії",
    )
    axes[0, 0].set_xlabel("Індекс вимірювання")
    axes[0, 0].set_ylabel("Ціна, USD")
    axes[0, 0].set_title("Виявлені аномалії на часовому ряді ціни")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Зміна ціни vs Зміна обсягу
    axes[0, 1].scatter(
        normal["price_change"],
        normal["volume_change"],
        c="blue",
        alpha=0.3,
        s=5,
        label="Нормальні",
    )
    axes[0, 1].scatter(
        anomalies["price_change"],
        anomalies["volume_change"],
        c="red",
        alpha=0.8,
        s=30,
        marker="x",
        label="Аномалії",
    )
    axes[0, 1].set_xlabel("Зміна ціни, %")
    axes[0, 1].set_ylabel("Зміна обсягу, %")
    axes[0, 1].set_title("Простір ознак: зміна ціни vs зміна обсягу")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Гістограма змін ціни для аномалій
    if len(anomalies) > 0:
        axes[1, 0].hist(
            anomalies["price_change"],
            bins=30,
            facecolor="red",
            alpha=0.6,
            edgecolor="black",
        )
        axes[1, 0].axvline(x=0, color="black", linestyle="--", linewidth=1)
        axes[1, 0].set_xlabel("Зміна ціни, %")
        axes[1, 0].set_ylabel("Частота")
        axes[1, 0].set_title("Розподіл змін ціни для аномальних точок")
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Відхилення від тренду vs волатильність
    axes[1, 1].scatter(
        normal["price_deviation"],
        normal["volatility"],
        c="blue",
        alpha=0.3,
        s=5,
        label="Нормальні",
    )
    axes[1, 1].scatter(
        anomalies["price_deviation"],
        anomalies["volatility"],
        c="red",
        alpha=0.8,
        s=30,
        marker="x",
        label="Аномалії",
    )
    axes[1, 1].set_xlabel("Відхилення від ковзного середнього, %")
    axes[1, 1].set_ylabel("Волатильність, USD")
    axes[1, 1].set_title("Відхилення від тренду vs волатильність")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        "DBSCAN: Виявлення аномалій в даних Bitcoin", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_dbscan.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("  КЛАСТЕРИЗАЦІЯ ДАНИХ BITCOIN (CoinGecko)")
    print("=" * 70)

    # -------------------------------------------------------------------------
    #  ЕТАП 1. Завантаження даних
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ЕТАП 1. Завантаження даних")
    print("=" * 70)

    csv_path = "bitcoin_prices_last_90_days.csv"
    df = pd.read_csv(csv_path)
    print(f"  Завантажено {len(df)} точок даних з {csv_path}")
    print(f"  Стовпці: {', '.join(df.columns)}")

    # -------------------------------------------------------------------------
    #  ЕТАП 2. Підготовка ознак
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ЕТАП 2. Підготовка ознак для кластеризації")
    print("=" * 70)

    df_features = prepare_features(df, window_size=24)

    # -------------------------------------------------------------------------
    #  ЕТАП 3. K-Means кластеризація
    # -------------------------------------------------------------------------
    df_features, kmeans, scaler_kmeans, inertias, silhouettes, K_range = (
        kmeans_clustering(
            df_features,
            n_clusters=3,
            features_list=["price_usd", "volume_usd", "volatility", "price_change"],
        )
    )

    # -------------------------------------------------------------------------
    #  ЕТАП 4. DBSCAN кластеризація
    # -------------------------------------------------------------------------
    df_features, dbscan, scaler_dbscan = dbscan_clustering(
        df_features,
        eps=0.8,
        min_samples=5,
        features_list=["price_change", "volume_change", "price_deviation"],
    )

    # -------------------------------------------------------------------------
    #  ЕТАП 5. Візуалізація
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ЕТАП 5. Візуалізація результатів")
    print("=" * 70)

    plot_kmeans_results(df_features, inertias, silhouettes, K_range)
    plot_dbscan_results(df_features)

    # -------------------------------------------------------------------------
    #  ЕТАП 6. Збереження результатів
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ЕТАП 6. Збереження результатів")
    print("=" * 70)

    output_file = "bitcoin_clustering_results.csv"
    df_features.to_csv(output_file, index=False)
    print(f"  Результати збережено в: {output_file}")

    # Підсумок
    print("\n" + "=" * 70)
    print("  ПІДСУМОК")
    print("=" * 70)
    print(f"""
  K-MEANS:
    - Виявлено 4 стани ринку (кластери)
    - Якість кластеризації (Silhouette): {silhouette_score(scaler_kmeans.fit_transform(df_features[["price_usd", "volume_usd", "volatility", "price_change"]].values), df_features["kmeans_cluster"]):.4f}

  DBSCAN:
    - Виявлено аномалій: {(df_features["dbscan_cluster"] == -1).sum()} ({(df_features["dbscan_cluster"] == -1).sum() / len(df_features) * 100:.2f}%)
    - Аномалії характеризуються різкими змінами ціни/обсягу

  Результати збережено:
    - bitcoin_clustering_results.csv
    - plot_clustering_kmeans.png
    - plot_clustering_dbscan.png
    """)
