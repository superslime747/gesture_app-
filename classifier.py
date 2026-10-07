"""
Модуль классификации жестов.
Загружает обученную модель (sklearn) из models/gesture_model.pkl
и предсказывает жест по landmarks.
"""
import os
import numpy as np
import joblib
from pathlib import Path

from camera import normalize_landmarks

# Путь к модели
MODEL_PATH = Path(__file__).parent / "models" / "gesture_model.pkl"

# Названия жестов (индекс = метка класса)
GESTURE_NAMES = {
    0: "Камень",
    1: "Ножницы",
    2: "Бумага",
    3: "Указательный",
    4: "Щипок",
    5: "Большой палец",
    6: "OK",
}

# Порог уверенности — ниже него жест считается "неизвестным"
CONFIDENCE_THRESHOLD = 0.6


class GestureClassifier:
    """Обёртка над sklearn-моделью для предсказания жестов."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                print(f"[classifier] Модель загружена: {self.model_path}")
            except Exception as e:
                print(f"[classifier] Ошибка загрузки модели: {e}")
                self.model = None
        else:
            print(f"[classifier] Модель не найдена: {self.model_path}")
            print("[classifier] Соберите данные (data_collector.py) и обучите модель (train.py).")

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, landmarks: np.ndarray):
        """
        Возвращает (label, name, confidence) или (None, "Неизвестно", 0.0).
        landmarks — np.array (21,3) в нормализованных координатах MediaPipe.
        """
        if self.model is None or landmarks is None:
            return None, "Неизвестно", 0.0

        features = normalize_landmarks(landmarks)
        if features is None:
            return None, "Неизвестно", 0.0
        features = features.reshape(1, -1)

        try:
            proba = self.model.predict_proba(features)[0]
            idx = int(np.argmax(proba))
            conf = float(proba[idx])
            if conf < CONFIDENCE_THRESHOLD:
                return None, "Неизвестно", conf
            label = int(self.model.classes_[idx])
            return label, GESTURE_NAMES.get(label, f"Класс {label}"), conf
        except Exception as e:
            print(f"[classifier] Ошибка предсказания: {e}")
            return None, "Неизвестно", 0.0