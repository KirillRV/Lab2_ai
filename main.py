# lab1_lab2_final_fixed.py
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# === Пути ===
INPUT_CSV = r'C:\Users\Kirill\PycharmProjects\lab1\train.csv'
OUT_DIR = r'C:\Users\Kirill\PycharmProjects\lab1\lab_results'
FINAL_CSV = r'C:\Users\Kirill\PycharmProjects\lab1\processed_Titanic-Dataset.csv'
os.makedirs(OUT_DIR, exist_ok=True)

# === Загрузка ===
try:
    df = pd.read_csv(INPUT_CSV)
    print("Датасет успешно загружен:", df.shape)
except Exception as e:
    print("Ошибка при загрузке CSV:", e)
    raise SystemExit

# === Добавляем искусственные пропуски во все столбцы ===
print("\nДобавляем пропуски во все столбцы для демонстрации...")
for col in df.columns:
    df.loc[df.sample(frac=0.02, random_state=42).index, col] = np.nan  # 2% NaN
print("Пропуски добавлены во все столбцы.\n")

print("Количество пропусков по каждому столбцу:")
print(df.isnull().sum())

# === Извлекаем дополнительные признаки ===
def extract_deck(cabin):
    if pd.isna(cabin): return 'Unknown'
    return str(cabin)[0]

def extract_title(name):
    if pd.isna(name): return 'Unknown'
    parts = str(name).split(',')
    if len(parts) > 1:
        return parts[1].split('.')[0].strip()
    return 'Unknown'

df['Deck'] = df['Cabin'].apply(extract_deck)
df['Title'] = df['Name'].apply(extract_title)

# === Заполнение пропусков ===
print("\nЗаполняем пропуски...")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

for c in num_cols:
    median = df[c].median()
    df[c] = df[c].fillna(median)
    print(f"  Числовая колонка {c} → медиана = {median}")

for c in cat_cols:
    mode = df[c].mode(dropna=True)
    fill = mode[0] if not mode.empty else 'Unknown'
    df[c] = df[c].fillna(fill)
    print(f"  Категориальная колонка {c} → мода = {fill}")

print("\nПропуски после заполнения:")
print(df.isnull().sum())

# === Нормализация числовых признаков ===
print("\nНормализуем числовые признаки (MinMax [0,1])...")
scaler = MinMaxScaler()
num_all = df.select_dtypes(include=[np.number]).columns
df[num_all] = scaler.fit_transform(df[num_all])
print("Нормализация завершена.")

# === One-Hot Encoding для категориальных признаков ===
print("\nПреобразуем категориальные признаки в числовые (OHE)...")
for_ohe = [c for c in cat_cols if c not in ['Name','Ticket','Cabin']]
df = pd.get_dummies(df, columns=for_ohe, drop_first=False)
print("OHE завершён. Размерность:", df.shape)

# === Сохраняем обработанный датасет ===
processed_path = os.path.join(OUT_DIR, 'processed_Titanic-Dataset_full.csv')
df.to_csv(processed_path, index=False)
print(f"\nОбработанный датасет сохранён: {processed_path}")

df.to_csv(FINAL_CSV, index=False)
print(f"Итоговый обработанный датасет сохранён как: '{FINAL_CSV}'")

# === Подготовка для ЛР2 ===
if 'Survived' not in df.columns or 'Fare' not in df.columns:
    raise SystemExit("Нет колонок Survived/Fare для выполнения ЛР2.")

X = df.drop(['Survived', 'Fare'], axis=1)
y_clf = df['Survived']
y_reg = df['Fare']

# ⚠️ Удаляем все оставшиеся текстовые колонки перед обучением
drop_cols = [c for c in X.columns if X[c].dtype == 'object']
if drop_cols:
    print("\n⚠️ Найдены строковые колонки, удаляем перед обучением:", drop_cols)
    X = X.drop(columns=drop_cols)

# Также удалим исходные поля, которые могли попасть после OHE
for col in ['Name', 'Ticket', 'Cabin']:
    if col in X.columns:
        X = X.drop(columns=[col], errors='ignore')

# Разделение выборки
X_train, X_test, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.3, random_state=42, stratify=y_clf
)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.3, random_state=42
)

# На случай NaN после OHE
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
X_train_reg = X_train_reg.fillna(0)
X_test_reg = X_test_reg.fillna(0)

print("\nРазделение данных выполнено.")
print("Train:", X_train.shape, "Test:", X_test.shape)

# === Регрессия (Fare) ===
print("\nЗадача регрессии (предсказание Fare)")
reg = LinearRegression()
reg.fit(X_train_reg, y_train_reg)
y_pred_reg = reg.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"  MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

plt.figure(figsize=(6,6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6)
plt.plot([0,1],[0,1],'--', color='red')
plt.title('Regression: Actual vs Predicted (Fare)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.savefig(os.path.join(OUT_DIR, 'regression_plot.png'))
plt.close()

# === Классификация (Survived) ===
print("\nЗадача классификации (предсказание Survived)")
clf = LogisticRegression(max_iter=1000, solver='liblinear')
clf.fit(X_train, y_train_clf)
y_pred_clf = clf.predict(X_test)

acc = accuracy_score(y_test_clf, y_pred_clf)
prec = precision_score(y_test_clf, y_pred_clf, zero_division=0)
rec = recall_score(y_test_clf, y_pred_clf, zero_division=0)
f1 = f1_score(y_test_clf, y_pred_clf, zero_division=0)
print(f"  Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

cm = confusion_matrix(y_test_clf, y_pred_clf)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Survived)')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix.png'))
plt.close()

# === Сохраняем краткий отчёт ===
with open(os.path.join(OUT_DIR, 'summary.txt'), 'w', encoding='utf-8') as f:
    f.write("Лабораторные 1 и 2 — Titanic Dataset\n")
    f.write("Все столбцы содержали искусственные ошибки (NaN), заполнены автоматически.\n")
    f.write("Выполнены задачи регрессии и классификации.\n")
    f.write(f"Regression: MSE={mse:.4f}, R2={r2:.4f}\n")
    f.write(f"Classification: Accuracy={acc:.4f}, F1={f1:.4f}\n")
    f.write("Результаты сохранены в lab_results и processed_Titanic-Dataset.csv\n")

print("\nОтчёт сохранён в summary.txt")
print("Работа полностью завершена. Все результаты — в папке lab_results/ и processed_Titanic-Dataset.csv")
