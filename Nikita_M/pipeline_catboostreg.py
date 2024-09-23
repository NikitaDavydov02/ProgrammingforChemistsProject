import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV

train_df = pd.read_csv('train_df.csv')
val_df = pd.read_csv('val_df.csv')

print(f"Train set: {train_df.shape}")
print(f"Validation set: {val_df.shape}")

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Определите целевую переменную и признаки
X = train_df.drop(['logD','smiles'], axis=1)
y = train_df['logD']

#Проверка валидационного сета
X_val = val_df.drop(['logD','smiles'], axis=1)
y_val = val_df['logD']

# Разделите данные на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=34)

# Создайте и обучите модель CatBoostRegressor
model = CatBoostRegressor(iterations=1000,       # Количество итераций
                          learning_rate=0.1,    # Шаг обучения
                          depth=6,              # Глубина дерева
                          eval_metric='RMSE',   # Оценка качества модели (RMSE)
                          verbose=100)          # Печать прогресса каждые 100 итераций

model.fit(X_train, y_train)

# Прогнозирование на тестовом наборе
y_pred = model.predict(X_test)
y1_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

print(f'r2_test : {metrics.r2_score(y_test, y_pred)}')
print(f'rmse_test : {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}')
print(f'r2_train : {metrics.r2_score(y_train, y1_pred)}')
print(f'rmse_train : {np.sqrt(metrics.mean_squared_error(y_train, y1_pred))}')
print(f'r2_val : {metrics.r2_score(y_val, y_val_pred)}')
print(f'rmse_val : {np.sqrt(metrics.mean_squared_error(y_val, y_val_pred))}')
plt.scatter(y_test, y_pred, color = 'blue')
plt.scatter(y_train, y1_pred, color = 'red')
plt.plot(y_train, y_train, color = 'black')
plt.plot(y_test, y_test, color = 'black')
plt.title('CatBoost')
plt.xlabel('test data, logD')
plt.ylabel('predicted data, logD')

plt.show()

# Получаем важность признаков

feature_importance = model.get_feature_importance()

# Создаем DataFrame с признаками и их важностью
feature_importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': feature_importance})

# Сортируем DataFrame по убыванию важности
feature_importance_df = feature_importance_df.sort_values('importance', ascending=True)

# Выбираем 0 худших признаков
worst_features = feature_importance_df.head(12)

# Выводим результат
print(worst_features)

columns_to_remove = worst_features.iloc[:, 0].tolist()

# Удаляем указанные столбцы из DataFrame df
train_df_cleaned = train_df.drop(columns=columns_to_remove, errors='ignore')
val_df_cleaned = val_df.drop(columns=columns_to_remove, errors='ignore')

# Определите целевую переменную и признаки
X = train_df_cleaned.drop(['logD','smiles'], axis=1)
y = train_df_cleaned['logD']

#Проверка валидационного сета
X_val = val_df_cleaned.drop(['logD','smiles'], axis=1)
y_val = val_df_cleaned['logD']

# Разделите данные на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=333)

# Создайте и обучите модель CatBoostRegressor
model = CatBoostRegressor(iterations=1000,       # Количество итераций
                          learning_rate=0.1,    # Шаг обучения
                          depth=6,              # Глубина дерева
                          eval_metric='RMSE',   # Оценка качества модели (RMSE)
                          l2_leaf_reg=7,  # Регуляризация L2 (только для деревьев)
                          early_stopping_rounds=600,
                          verbose=100)          # Печать прогресса каждые 100 итераций
model.fit(X_train, y_train)

# Прогнозирование на тестовом наборе
y_pred = model.predict(X_test)
y1_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

print(f'r2_test : {metrics.r2_score(y_test, y_pred)}')
print(f'rmse_test : {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}')
print(f'r2_train : {metrics.r2_score(y_train, y1_pred)}')
print(f'rmse_train : {np.sqrt(metrics.mean_squared_error(y_train, y1_pred))}')
print(f'r2_val : {metrics.r2_score(y_val, y_val_pred)}')
print(f'rmse_val : {np.sqrt(metrics.mean_squared_error(y_val, y_val_pred))}')
plt.scatter(y_test, y_pred, color = 'blue')
plt.scatter(y_train, y1_pred, color = 'red')
plt.plot(y_train, y_train, color = 'black')
plt.plot(y_test, y_test, color = 'black')
plt.title('CatBoost')
plt.xlabel('test data, logD')
plt.ylabel('predicted data, logD')

plt.show()
