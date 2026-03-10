#!/bin/bash
echo "Шаг 1: Создание данных"
python3 data_creation.py

echo "Шаг 2: Предобработка данных"
python3 model_preprocessing.py

echo "Шаг 3: Обучение модели"
python3 model_preparation.py

echo "Шаг 14: Тестирование модели"
python3 model_testing.py

echo "THe End."

