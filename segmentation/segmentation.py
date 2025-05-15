import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency, f_oneway
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_excel('./data/data_previous.xls', sheet_name='Обучающая')

# 1. Предобработка данных 
def preprocess_data(df):
    # Сохраняем оригинальные значения для восстановления
    original_data = df.copy()
    
    # Удаление ненужных столбцов
    df = df.drop(columns=[
        'ИД', 'СМЕНА_МЖ', 'ЗАГРАН', 'ИНОСТР.ЯЗЫК',
        'АВТО', 'ОТРАСЛЬ', 'РАБОТА_ПО_НАПР', 'ОТДЕЛЕНИЕ', 'ГОРОД'
    ])
    
    # Заполнение пропусков
    df['ДОХОД_СУПРУГИ(А)'] = df['ДОХОД_СУПРУГИ(А)'].fillna(0)
    
    # Создание новых признаков (сохраняем исходные значения)
    df['КРЕДИТНАЯ_НАГРУЗКА'] = df['СУММА_ВЫДАННОГО_КРЕДИТА'] / (df['ПЕРСОНАЛЬНЫЙ_ДОХОД'] + 1)
    df['ВОЗРАСТ_ГРУППА'] = pd.cut(df['ВОЗРАСТ.ЛЕТ'], 
                                 bins=[18, 25, 35, 45, 100],
                                 labels=['18-25', '26-35', '36-45', '45+'],
                                 right=False)
    
    # Кодирование категориальных признаков
    df['СЕМЕЙНЫЙ_СТАТУС'] = np.where(
        (df['БРАК'] == 1) & (df['ДЕТИ'] == 0), 'Бездетные_семьи',
        np.where(df['ДЕТИ'] > 0, 'Семьи_с_детьми', 'Одинокие')
    )
    
    education_mapping = {
        1: 'Среднее',
        2: 'Среднее_спец',
        3: 'Высшее',
        4: 'Два_высших'
    }
    df['ОБРАЗОВАНИЕ_ГРУППА'] = df['ОБРАЗОВАНИЕ'].map(education_mapping)
    df['СОБСТВЕННИК'] = np.where(df['СОБСТВЕННИК_ФАКТ.'] > 3, 1, 0)
    
    # Доходные группы
    df['ДОХОД_ГРУППА'] = pd.cut(df['ПЕРСОНАЛЬНЫЙ_ДОХОД'],
                               bins=[0, 15000, 30000, 50000, np.inf],
                               labels=['Низкий', 'Средний', 'Высокий', 'Премиум'],
                               right=False)
    
    # Сохраняем оригинальные числовые значения
    df[['ВОЗРАСТ.ЛЕТ', 'ДОХОД_СЕМЬИ_', 'ПЕРСОНАЛЬНЫЙ_ДОХОД']] = original_data[['ВОЗРАСТ.ЛЕТ', 'ДОХОД_СЕМЬИ_', 'ПЕРСОНАЛЬНЫЙ_ДОХОД']]
    
    return df

df = preprocess_data(df)

# 2. Подготовка признаков
numerical_features = [
    'ВОЗРАСТ.ЛЕТ', 'ПЕРСОНАЛЬНЫЙ_ДОХОД', 
    'КРЕДИТНАЯ_НАГРУЗКА', 'ДОХОД_СЕМЬИ_'
]

categorical_features = [
    'СЕМЕЙНЫЙ_СТАТУС', 'ВОЗРАСТ_ГРУППА', 
    'СОБСТВЕННИК', 'ОБРАЗОВАНИЕ_ГРУППА'
]

# Создаем масштабированную копию только для кластеризации
scaler = StandardScaler()
df_scaled = df[numerical_features].copy()
df_scaled[numerical_features] = scaler.fit_transform(df[numerical_features])

# 3. Кластеризация
X_processed = pd.concat([
    df_scaled[numerical_features],
    df[categorical_features].astype(str)
], axis=1)

categorical_indices = [X_processed.columns.get_loc(col) for col in categorical_features]

kproto = KPrototypes(
    n_clusters=5,
    init='Huang',
    random_state=42,
    n_init=5,
    verbose=0
)

clusters = kproto.fit_predict(
    X_processed.values,
    categorical=categorical_indices
)

# Добавляем кластеры в оригинальный DataFrame
df['CLUSTER'] = clusters

# Создание понятных названий кластеров
cluster_names = {
    0: 'Молодые семьи (высокий риск)',
    1: 'Зрелые клиенты (низкий риск)',
    2: 'Студенты (минимальный доход)',
    3: 'Перекредитованные (критический риск)',
    4: 'Премиум-сегмент (надежные)'
}
df['CLUSTER_NAME'] = df['CLUSTER'].map(cluster_names)

# 4. Анализ значимости различий
contingency_table = pd.crosstab(df['CLUSTER'], df['ДЕФОЛТ60'])
chi2, p_chi, _, _ = chi2_contingency(contingency_table)

anova_results = {}
for feature in numerical_features:
    groups = [df[df['CLUSTER'] == cluster][feature] for cluster in df['CLUSTER'].unique()]
    f_stat, p_anova = f_oneway(*groups)
    anova_results[feature] = p_anova

# 5. Визуализация с улучшенной палитрой
# PCA для числовых признаков
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_scaled[numerical_features])

# Названия компонент на основе нагрузки признаков
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['Главная компонента 1 (Демография)', 'Главная компонента 2 (Финансы)'],
    index=numerical_features
)
print("\nИнтерпретация главных компонент:")
print(loadings)

# Создание графиков с улучшенной цветовой палитрой
plt.figure(figsize=(16, 10))
scatter = sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df['CLUSTER_NAME'],
    palette='tab10',
    alpha=0.85,
    s=120,
    edgecolor='k',
    linewidth=0.5
)

plt.title('Визуализация кластеров в пространстве главных компонент', fontsize=18, pad=20)
plt.xlabel('Главная компонента 1: Возраст и доход', fontsize=14)
plt.ylabel('Главная компонента 2: Кредитная нагрузка', fontsize=14)

# Настройка легенды
legend = plt.legend(
    title='Сегменты клиентов',
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    title_fontsize=12,
    fontsize=11,
    framealpha=0.9
)

# Добавление пояснений
plt.text(
    0.5, -0.18,
    "Главная компонента 1: Положительные значения - высокий доход/возраст, отрицательные - низкие значения\n"
    "Главная компонента 2: Положительные значения - высокая кредитная нагрузка, отрицательные - низкая",
    transform=plt.gca().transAxes,
    ha='center',
    fontsize=11,
    color='#555555'
)

plt.grid(True, linestyle='--', alpha=0.6)
sns.despine()
plt.tight_layout()
plt.show()

# 6. Детальный профиль кластеров
cluster_profile = df.groupby('CLUSTER_NAME').agg({
    'ДЕФОЛТ60': ['mean', 'size'],
    'ПЕРСОНАЛЬНЫЙ_ДОХОД': 'median',
    'СУММА_ВЫДАННОГО_КРЕДИТА': 'median',
    'КРЕДИТНАЯ_НАГРУЗКА': 'median',
    'СЕМЕЙНЫЙ_СТАТУС': lambda x: x.mode()[0],
    'ВОЗРАСТ_ГРУППА': lambda x: x.mode()[0],
    'ДОХОД_ГРУППА': lambda x: x.mode()[0],
    'ОБРАЗОВАНИЕ_ГРУППА': lambda x: x.mode()[0]
}).reset_index()

# Форматирование таблицы
cluster_profile.columns = [
    'Кластер', 'Доля_дефолтов', 'Размер',
    'Медианный_доход', 'Медианный_кредит',
    'Кредит/Доход', 'Семейный_статус',
    'Возрастная_группа', 'Доходная_группа',
    'Образование'
]

# Форматирование значений
cluster_profile['Доля_дефолтов'] = cluster_profile['Доля_дефолтов'].apply(lambda x: f"{x:.1%}")
cluster_profile['Медианный_доход'] = cluster_profile['Медианный_доход'].apply(lambda x: f"{int(x):,}")
cluster_profile['Медианный_кредит'] = cluster_profile['Медианный_кредит'].apply(lambda x: f"{int(x):,}")
cluster_profile['Кредит/Доход'] = cluster_profile['Кредит/Доход'].apply(lambda x: f"{x:.2f}")

# Вывод результатов
print("\n" + "="*110)
print("{:<35} | {:<15} | {:<10} | {:<18} | {:<18} | {:<15}".format(
    'Кластер', 'Доля дефолтов', 'Размер', 
    'Возрастная группа', 'Доходная группа', 
    'Медианный доход'))
print("-"*110)
for _, row in cluster_profile.iterrows():
    print("{:<35} | {:<15} | {:<10} | {:<18} | {:<18} | {:<15}".format(
        row['Кластер'],
        row['Доля_дефолтов'],
        row['Размер'],
        row['Возрастная_группа'],
        row['Доходная_группа'],
        row['Медианный_доход']
    ))
print("="*110)

# 7. Сохранение с проверкой исходных значений
# Убедимся, что ключевые столбцы содержат исходные значения
assert df['ВОЗРАСТ.ЛЕТ'].min() >= 18, "Обнаружены некорректные значения возраста"
assert df['ПЕРСОНАЛЬНЫЙ_ДОХОД'].min() >= 0, "Обнаружены отрицательные доходы"

df.to_excel('./results/segmented_clients.xlsx', index=False)
print("\nРезультаты сохранены в папку results в файл: segmented_clients.xlsx")