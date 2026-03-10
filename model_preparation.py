import pandas as pd
import os
import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib


def prepare_and_train_model():

    print("Начало обучения модели")

    train_files = glob.glob("train/preprocessed/*.csv")

    if not train_files:
        print("Ошибка: Не найдены предобработанные тренировочные данные!")
        return

    all_data = []

    for file in train_files:
        df = pd.read_csv(file)
        all_data.append(df)
        print(f"Загружен файл: {os.path.basename(file)}")

    train_data = pd.concat(all_data, ignore_index=True)

    feature_columns = ['dayofweek', 'month', 'day', 'close_lag1', 'close_lag7', 'close_ma5', 'close_ma20']

    available_features = [col for col in feature_columns if col in train_data.columns]

    X = train_data[available_features]
    y = train_data['close']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nРазмер обучающей выборки: {X_train.shape}")
    print(f"Размер валидационной выборки: {X_val.shape}")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    print("\nОбучение модели...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"\nМетрики на валидационной выборке:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    joblib.dump(model, 'trained_model.pkl')
    print("\nМодель сохранена как 'trained_model.pkl'")

    return model


if __name__ == "__main__":
    prepare_and_train_model()