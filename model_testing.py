import pandas as pd
import os
import glob
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def test_model():

    print("Начало тестирования модели")

    if not os.path.exists('trained_model.pkl'):
        print("Ошибка: Модель не найдена!")
        return

    model = joblib.load('trained_model.pkl')
    print("Модель загружена")

    if os.path.exists('scaler.pkl'):
        scaler = joblib.load('scaler.pkl')
        print("Scaler загружен")
    else:
        scaler = None
        print("Предупреждение: Scaler не найден")

    test_files = glob.glob("test/preprocessed/*.csv")

    if not test_files:
        print("Ошибка: Не найдены предобработанные тестовые данные!")
        return

    feature_columns = ['dayofweek', 'month', 'day', 'close_lag1', 'close_lag7', 'close_ma5', 'close_ma20']

    all_predictions = []
    all_actual = []

    print(f"\nНайдено тестовых файлов: {len(test_files)}")

    for file in test_files:
        df = pd.read_csv(file)

        available_features = [col for col in feature_columns if col in df.columns]

        if len(available_features) < len(feature_columns):
            print(f"Предупреждение в файле {os.path.basename(file)}: Не все признаки доступны")

        X_test = df[available_features]
        y_test = df['close']

        y_pred = model.predict(X_test)

        file_predictions = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'file': os.path.basename(file)
        })

        all_predictions.append(file_predictions)
        all_actual.extend(y_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\nФайл: {os.path.basename(file)}")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2 Score: {r2:.4f}")

    if all_predictions:
        all_results = pd.concat(all_predictions, ignore_index=True)

        overall_mse = mean_squared_error(all_results['actual'], all_results['predicted'])
        overall_mae = mean_absolute_error(all_results['actual'], all_results['predicted'])
        overall_r2 = r2_score(all_results['actual'], all_results['predicted'])

        print("\n" + "=" * 50)
        print("ОБЩИЕ МЕТРИКИ НА ВСЕХ ТЕСТОВЫХ ДАННЫХ:")
        print(f"MSE: {overall_mse:.4f}")
        print(f"MAE: {overall_mae:.4f}")
        print(f"R2 Score: {overall_r2:.4f}")
        print("=" * 50)

if __name__ == "__main__":
    test_model()