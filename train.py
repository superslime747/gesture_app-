"""
Скрипт обучения классификатора жестов.
Читает data/gestures.csv, обучает MLPClassifier, сохраняет модель.
"""
import csv
from pathlib import Path

import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA_PATH = Path(__file__).parent / "data" / "gestures.csv"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "gesture_model.pkl"


def load_data():
    """Читает CSV и возвращает X, y."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Файл данных не найден: {DATA_PATH}. "
                                f"Сначала запустите data_collector.py")

    X, y = [], []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # пропускаем заголовок
        for row in reader:
            if len(row) < 2:
                continue
            try:
                feats = [float(v) for v in row[:63]]
                label = int(row[63])
            except ValueError:
                continue
            X.append(feats)
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main():
    print("=" * 60)
    print("ОБУЧЕНИЕ КЛАССИФИКАТОРА ЖЕСТОВ")
    print("=" * 60)

    X, y = load_data()
    print(f"Загружено примеров: {len(X)}")
    print(f"Классы: {sorted(set(y.tolist()))}")

    if len(X) < 50:
        print("Слишком мало данных для обучения. Соберите больше примеров.")
        return

    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # MLP — простая полносвязная сеть
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True,
    )

    print("Обучение модели...")
    model.fit(X_train, y_train)

    # Оценка
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy на тесте: {acc:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Сохраняем модель
    joblib.dump(model, MODEL_PATH)
    print(f"\nМодель сохранена: {MODEL_PATH}")


if __name__ == "__main__":
    main()