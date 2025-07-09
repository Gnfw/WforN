import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings

# Отключение предупреждений
warnings.filterwarnings('ignore')

# Настройка отображения графиков
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 6)
pd.set_option('display.max_columns', None)

def load_and_preprocess_data(file_path):
    """Загрузка и предварительная обработка данных"""
    try:
        df = pd.read_csv(file_path)
        
        # Проверка необходимых столбцов
        required_columns = ['country', 'points', 'price', 'variety', 'province']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Отсутствует обязательный столбец: {col}")
        
        # Обработка пропусков
        df['price'].fillna(df['price'].median(), inplace=True)
        df.dropna(subset=required_columns, inplace=True)
        
        # Преобразование типов
        df['price'] = df['price'].astype(float)
        
        # Удаление дубликатов
        df.drop_duplicates(inplace=True)
        
        return df
    
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
        return None

def analyze_oceania_wines(df):
    """Анализ вин Океании"""
    # Фильтрация данных по Океании
    oceania = df[df['country'].isin(['Australia', 'New Zealand'])]
    
    if oceania.empty:
        print("Нет данных по винам Океании")
        return
    
    print("\n=== Общий анализ вин Океании ===")
    
    # 1. Популярные сорта вин (Топ-5)
    print("\nТоп-5 популярных сортов вин в Океании:")
    top_varieties = oceania['variety'].value_counts().head(5)
    print(top_varieties)
    
    # Визуализация
    plt.figure(figsize=(10, 5))
    top_varieties.plot(kind='bar', color='skyblue')
    plt.title('Топ-5 популярных сортов вин в Океании')
    plt.xlabel('Сорт вина')
    plt.ylabel('Количество')
    plt.xticks(rotation=45)
    plt.show()
    
    # 2. Статистика цен по странам
    print("\nСтатистика цен по странам:")
    price_stats = oceania.groupby('country')['price'].agg(['mean', 'max', 'min', 'count'])
    print(price_stats)
    
    # 3. Лучшие вина по рейтингу (Топ-10)
    print("\nТоп-10 вин Океании по рейтингу:")
    top_wines = oceania.sort_values('points', ascending=False).head(10)[['country', 'variety', 'points', 'price', 'winery']]
    print(top_wines)
    
    # 4. Зависимость цены от рейтинга
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='points', y='price', data=oceania, hue='country')
    plt.title('Зависимость цены от рейтинга в Океании')
    plt.xlabel('Рейтинг')
    plt.ylabel('Цена ($)')
    plt.show()
    
    # Корреляция между ценой и рейтингом
    corr = oceania['points'].corr(oceania['price'])
    print(f"\nКорреляция между рейтингом и ценой: {corr:.2f}")
    
    # 5. Распределение рейтингов по странам
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='country', y='points', data=oceania)
    plt.title('Распределение рейтингов по странам Океании')
    plt.xlabel('Страна')
    plt.ylabel('Рейтинг')
    plt.show()
    
    return oceania

def analyze_sauvignon_blanc(oceania_data):
    """Углубленный анализ Sauvignon Blanc"""
    sauv_blanc = oceania_data[oceania_data['variety'] == 'Sauvignon Blanc']
    
    if sauv_blanc.empty:
        print("\nНет данных по Sauvignon Blanc в Океании")
        return
    
    print("\n=== Углубленный анализ Sauvignon Blanc ===")
    
    # 1. Основная статистика
    print("\nОсновная статистика по Sauvignon Blanc:")
    print(f"Всего образцов: {len(sauv_blanc)}")
    print(f"Средний рейтинг: {sauv_blanc['points'].mean():.1f}")
    print(f"Средняя цена: ${sauv_blanc['price'].mean():.2f}")
    
    # 2. Сравнение по странам
    print("\nСравнение Sauvignon Blanc по странам:")
    country_stats = sauv_blanc.groupby('country')['points'].agg(['mean', 'count'])
    print(country_stats)
    
    # Визуализация
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='country', y='points', data=sauv_blanc)
    plt.title('Сравнение рейтингов Sauvignon Blanc по странам')
    plt.xlabel('Страна')
    plt.ylabel('Рейтинг')
    plt.show()
    
    # 3. Распределение цен
    plt.figure(figsize=(10, 6))
    sns.histplot(sauv_blanc['price'], bins=20, kde=True)
    plt.title('Распределение цен на Sauvignon Blanc в Океании')
    plt.xlabel('Цена ($)')
    plt.ylabel('Количество')
    plt.show()
    
    # 4. Проверка гипотез
    print("\n=== Проверка гипотез ===")
    
    # Гипотеза 1: Различаются ли средние рейтинги между странами?
    aus = sauv_blanc[sauv_blanc['country'] == 'Australia']['points']
    nz = sauv_blanc[sauv_blanc['country'] == 'New Zealand']['points']
    
    print("\nГипотеза 1: Средний рейтинг Sauvignon Blanc различается между Австралией и Новой Зеландией")
    t_stat, p_value = stats.ttest_ind(aus, nz, equal_var=False)
    print(f"Результат t-теста: p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("Вывод: Отвергаем нулевую гипотезу - рейтинги статистически значимо различаются")
        print(f"Средний рейтинг: Австралия = {aus.mean():.1f}, Новая Зеландия = {nz.mean():.1f}")
    else:
        print("Вывод: Нет оснований отвергать нулевую гипотезу - различия не статистически значимы")
    
    # Гипотеза 2: Есть ли корреляция между ценой и рейтингом?
    print("\nГипотеза 2: Существует корреляция между ценой и рейтингом для Sauvignon Blanc")
    corr, p_value = stats.pearsonr(sauv_blanc['price'], sauv_blanc['points'])
    print(f"Коэффициент корреляции Пирсона: {corr:.2f}")
    print(f"p-value: {p_value:.4f}")
    if p_value < 0.05:
        if corr > 0:
            print("Вывод: Есть слабая положительная корреляция - с ростом цены рейтинг немного увеличивается")
        else:
            print("Вывод: Есть слабая отрицательная корреляция")
    else:
        print("Вывод: Корреляция статистически не значима")

def main():
    print("Анализ вин Океании (Австралия и Новая Зеландия) с акцентом на Sauvignon Blanc")
    
    # Загрузка файла (замените путь на свой)
    file_path = input("Введите путь к CSV-файлу с данными о винах: ") or 'wine_reviews.csv'
    
    # Загрузка и предобработка данных
    df = load_and_preprocess_data(file_path)
    if df is None:
        return
    
    # Анализ вин Океании
    oceania_data = analyze_oceania_wines(df)
    
    if oceania_data is not None:
        # Углубленный анализ Sauvignon Blanc
        analyze_sauvignon_blanc(oceania_data)

if __name__ == "__main__":
    main()
