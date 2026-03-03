import requests
import pandas as pd
from datetime import datetime


def parse_coingecko_bitcoin_prices(days=90):
    api_url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

    params = {"vs_currency": "usd", "days": days}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }

    try:
        print("Завантаження даних про ціну Bitcoin з CoinGecko API...")
        response = requests.get(api_url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()

            # Витягуємо дані про ціни
            prices = data.get("prices")
            market_caps = data.get("market_caps")
            volumes = data.get("total_volumes")

            # Створюємо DataFrame
            df_data = []
            for i in range(len(prices)):
                timestamp = prices[i][0]
                price = prices[i][1]

                # Отримуємо капіталізацію та обсяг, якщо доступно
                market_cap = market_caps[i][1] if i < len(market_caps) else None
                volume = volumes[i][1] if i < len(volumes) else None

                df_data.append(
                    {
                        "timestamp": timestamp,
                        "date": datetime.fromtimestamp(timestamp / 1000).strftime(
                            "%Y-%m-%d"
                        ),
                        "datetime": datetime.fromtimestamp(timestamp / 1000).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "price_usd": round(price, 2),
                        "market_cap_usd": round(market_cap, 2) if market_cap else None,
                        "volume_usd": round(volume, 2) if volume else None,
                    }
                )

            df = pd.DataFrame(df_data)

            # Зберігаємо в CSV
            start_date = df["date"].min()
            end_date = df["date"].max()
            output_file = f"bitcoin_prices_last_{days}_days.csv"
            df.to_csv(output_file, index=False)

            print(f"\nУспішно завантажено {len(df)} точок даних")
            print(f"Дані збережено в: {output_file}")
            print("\n--- Підсумкова статистика ---")
            print(f"Діапазон дат: {start_date} до {end_date}")
            print(
                f"Діапазон цін: ${df['price_usd'].min():,.2f} - ${df['price_usd'].max():,.2f}"
            )
            print(f"Середня ціна: ${df['price_usd'].mean():,.2f}")
            print(f"Поточна ціна (остання): ${df['price_usd'].iloc[-1]:,.2f}")

            # Показуємо перші кілька рядків
            print("\n--- Перші 5 записів ---")
            print(df.head())

            return df

        else:
            print(f"API запит не вдався з кодом статусу: {response.status_code}")
            print(f"Відповідь: {response.text[:500]}")
            return None

    except Exception as e:
        print(f"Помилка завантаження даних: {e}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("Парсер цін Bitcoin з CoinGecko")
    print("=" * 60)
    print()

    # Завантажуємо дані за останні 90 днів
    df = parse_coingecko_bitcoin_prices(days=90)
