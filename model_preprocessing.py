import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import joblib
import glob


def preprocess_data():
    os.makedirs("train/preprocessed", exist_ok=True)
    os.makedirs("test/preprocessed", exist_ok=True)

    scaler = StandardScaler()

    train_files = glob.glob("train/*.csv")
    train_data = []

    print("Обработка тренировочных данных")
    for file in train_files:
        df = pd.read_csv(file)

        df['date'] = pd.to_datetime(df['date'])
        df['dayofweek'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day

        df['close_lag1'] = df['close'].shift(1)
        df['close_lag7'] = df['close'].shift(7)

        df['close_ma5'] = df['close'].rolling(window=5).mean()
        df['close_ma20'] = df['close'].rolling(window=20).mean()

        df = df.dropna()

        base_name = os.path.basename(file)
        name_without_ext = os.path.splitext(base_name)[0]
        df.to_csv(f"train/preprocessed/{name_without_ext}_preprocessed.csv", index=False)

        train_data.append(df)

        print(f"Обработан файл: {base_name}")

    if train_data:
        all_train_close = pd.concat([df['close'] for df in train_data])
        scaler.fit(all_train_close.values.reshape(-1, 1))

        joblib.dump(scaler, 'scaler.pkl')
        print("Scaler обучен и сохранен")

    test_files = glob.glob("test/*.csv")
    print("\nОбработка тестовых данных...")

    for file in test_files:
        df = pd.read_csv(file)

        df['date'] = pd.to_datetime(df['date'])
        df['dayofweek'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day

        df['close_lag1'] = df['close'].shift(1)
        df['close_lag7'] = df['close'].shift(7)

        df['close_ma5'] = df['close'].rolling(window=5).mean()
        df['close_ma20'] = df['close'].rolling(window=20).mean()

        df = df.dropna()

        if len(df) > 0:
            df['close_scaled'] = scaler.transform(df['close'].values.reshape(-1, 1))

        base_name = os.path.basename(file)
        name_without_ext = os.path.splitext(base_name)[0]
        df.to_csv(f"test/preprocessed/{name_without_ext}_preprocessed.csv", index=False)

        print(f"Обработан файл: {base_name}")


if __name__ == "__main__":
    print("Начало предобработки данных.")
    preprocess_data()
    print("Предобработка завершена!")