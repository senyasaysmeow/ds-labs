import geopandas as gpd
import pandas as pd
import folium
import json
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GISBackendSystem:
    """
    Макет Backend-компоненти GIS системи.
    Призначення: визначення, аналітична обробка та геопросторова візуалізація 
    інформації про щільність заселення територій для обраного регіону.
    """
    
    def __init__(self, target_region="Europe"):
        self.target_region = target_region
        self.gdf = None
        self.stats = {}
        logging.info(f"Ініціалізація GIS Backend для регіону: {self.target_region}")

    def fetch_and_prepare_data(self):
        """
        Завантаження геоданих (кордонів) та демографічної інформації.
        Для макету використовується вбудований набір даних 'naturalearth_lowres'.
        """
        logging.info("Завантаження геопросторових даних...")
        try:
            # Завантаження набору даних Natural Earth безпосередньо з архіву
            url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
            world = gpd.read_file(url)
            
            # Приведення імен колонок до потрібного формату (POP_EST -> pop_est, CONTINENT -> continent, NAME -> name)
            world = world.rename(columns={'POP_EST': 'pop_est', 'CONTINENT': 'continent', 'NAME': 'name'})
            
            # Фільтрація за обраним регіоном (континентом)
            self.gdf = world[world['continent'] == self.target_region].copy()
            
            if self.gdf.empty:
                raise ValueError(f"Регіон '{self.target_region}' не знайдено.")
                
            logging.info(f"Успішно завантажено дані для {len(self.gdf)} територіальних одиниць.")
        except Exception as e:
            logging.error(f"Помилка при завантаженні даних: {e}")
            raise

    def calculate_population_density(self):
        """
        Визначення щільності заселення (осіб на квадратний кілометр).
        Розрахунок відбувається з перепроєкцією у систему рівних площ (Equal-Area).
        """
        logging.info("Проведення розрахунку щільності населення...")
        
        # Перепроєкція в Equal-Area (EPSG:6933) для точного розрахунку площі в метрах
        gdf_proj = self.gdf.to_crs("EPSG:6933")
        
        # Обчислення площі в квадратних кілометрах (1 км² = 1 000 000 м²)
        self.gdf['area_sq_km'] = gdf_proj.geometry.area / 1e6
        
        # Обчислення щільності населення (осіб / км²)
        # Відкидаємо території з нульовою площею або нульовим населенням для уникнення помилок
        self.gdf = self.gdf[(self.gdf['area_sq_km'] > 0) & (self.gdf['pop_est'] > 0)].copy()
        self.gdf['pop_density'] = self.gdf['pop_est'] / self.gdf['area_sq_km']
        
        # Округлення для зручності
        self.gdf['pop_density'] = self.gdf['pop_density'].round(2)
        logging.info("Розрахунок щільності населення завершено успішно.")

    def analytical_processing(self):
        """
        Аналітична обробка розрахованих даних: 
        збір статистики, агрегація та категоризація регіонів.
        """
        logging.info("Початок аналітичної обробки...")
        
        # Категоризація щільності (Низька, Середня, Висока) на основі квантилів
        q33 = self.gdf['pop_density'].quantile(0.33)
        q66 = self.gdf['pop_density'].quantile(0.66)
        
        def categorize_density(val):
            if val <= q33: return 'Низька'
            elif val <= q66: return 'Середня'
            else: return 'Висока'
            
        self.gdf['density_category'] = self.gdf['pop_density'].apply(categorize_density)
        
        # Розрахунок загальних статистичних показників
        self.stats = {
            "total_population": int(self.gdf['pop_est'].sum()),
            "total_area_sq_km": float(self.gdf['area_sq_km'].sum()),
            "average_density": float(self.gdf['pop_density'].mean()),
            "max_density": float(self.gdf['pop_density'].max()),
            "min_density": float(self.gdf['pop_density'].min()),
            "most_dense_territory": self.gdf.loc[self.gdf['pop_density'].idxmax(), 'name'],
            "least_dense_territory": self.gdf.loc[self.gdf['pop_density'].idxmin(), 'name']
        }
        
        logging.info("Аналітичну обробку завершено.")
        return self.stats

    def generate_geospatial_visualization(self, output_file="population_density_map.html"):
        """
        Геопросторова візуалізація у вигляді інтерактивної Choropleth-карти.
        """
        logging.info("Генерація геопросторової візуалізації...")
        
        # Визначення центру мапи (в системі координат WGS84, EPSG:4326)
        # Використовуємо .to_crs("EPSG:4326") про всяк випадок, якщо координати змінилися
        gdf_wgs84 = self.gdf.to_crs("EPSG:4326")
        
        # Застосування warning ignore для центроїдів, бо shapely може скаржитись на географічні координати
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            center_lat = gdf_wgs84.geometry.centroid.y.mean()
            center_lon = gdf_wgs84.geometry.centroid.x.mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles="CartoDB positron")
        
        # Створення шару Choropleth
        choropleth = folium.Choropleth(
            geo_data=self.gdf.to_json(),
            name='Щільність населення',
            data=self.gdf,
            columns=['name', 'pop_density'],
            key_on='feature.properties.name',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Щільність населення (осіб на км²)',
            highlight=True
        ).add_to(m)
        
        # Додавання інтерактивних підказок (Tooltip)
        tooltip = folium.features.GeoJsonTooltip(
            fields=['name', 'pop_est', 'area_sq_km', 'pop_density', 'density_category'],
            aliases=['Територія:', 'Населення:', 'Площа (км²):', 'Щільність (осіб/км²):', 'Рівень щільності:'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px; border-radius: 3px; box-shadow: 3px 3px 3px rgba(0,0,0,0.2);"),
            localize=True
        )
        
        # Прив'язуємо tooltip до існуючого geojson в choropleth
        choropleth.geojson.add_child(tooltip)
        
        # Додавання контролера шарів
        folium.LayerControl().add_to(m)
        
        # Збереження результату
        m.save(output_file)
        logging.info(f"Візуалізацію збережено у файл: {output_file}")
        return output_file

def run_gis_pipeline():
    print("="*60)
    print(" R&D Лабораторія: Запуск макету GIS Backend системи")
    print("="*60)
    
    # Ініціалізуємо бекенд для Європи
    gis = GISBackendSystem(target_region="Europe")
    
    # 1. Визначення та підготовка
    gis.fetch_and_prepare_data()
    
    # 2. Розрахунок щільності
    gis.calculate_population_density()
    
    # 3. Аналітична обробка
    stats = gis.analytical_processing()
    
    # Вивід результатів
    print("\n--- Результати Аналітики ---")
    print(f"Обраний регіон: {gis.target_region}")
    print(f"Загальне населення: {stats['total_population']:,}".replace(',', ' '))
    print(f"Загальна площа: {stats['total_area_sq_km']:,.2f} км²".replace(',', ' '))
    print(f"Середня щільність: {stats['average_density']:.2f} осіб/км²")
    print(f"Найбільш густонаселена територія: {stats['most_dense_territory']} ({stats['max_density']:.2f} осіб/км²)")
    print(f"Найменш густонаселена територія: {stats['least_dense_territory']} ({stats['min_density']:.2f} осіб/км²)")
    print("----------------------------\n")
    
    # 4. Візуалізація
    map_file = "population_density_europe.html"
    gis.generate_geospatial_visualization(map_file)
    print(f"✅ Процес успішно завершено. Відкрийте файл '{map_file}' у браузері для перегляду результату.")

if __name__ == "__main__":
    run_gis_pipeline()
