"""
Модуль распознавания лиц и эмоций.

- Идентификация: по фотографиям в data/faces/<имя>/*.jpg
- Эмоции: 7 классов (angry, disgust, fear, happy, sad, surprise, neutral)
  через DeepFace.
- Кэш базы лиц: data/faces_db.pkl (создаётся автоматически).

Использование:
    fr = FaceRecognizer()
    name, emotion = fr.analyze(bgr_frame)
"""
# ---- Глушим логи TensorFlow ДО импорта ----
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# --------------------------------------------

import pickle
from pathlib import Path

import cv2
import numpy as np


# DeepFace импортируем лениво (тяжёлый импорт)
_DEEPFACE = None


def _get_deepface():
    global _DEEPFACE
    if _DEEPFACE is None:
        from deepface import DeepFace
        _DEEPFACE = DeepFace
    return _DEEPFACE


# ----------------------------------------------------------------------
# Пути
# ----------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
FACES_DIR = DATA_DIR / "faces"
DB_PATH = DATA_DIR / "faces_db.pkl"

FACES_DIR.mkdir(parents=True, exist_ok=True)

UNKNOWN_NAME = "Какой-то чел"

# ----------------------------------------------------------------------
# Перевод эмоций на русский
# ----------------------------------------------------------------------
EMOTION_RU = {
    "angry":    "Злость",
    "disgust":  "Отвращение",
    "fear":     "Страх",
    "happy":    "Радость",
    "sad":      "Грусть",
    "surprise": "Удивление",
    "neutral":  "Нейтрально",
}

# Эмодзи для отображения
EMOTION_EMOJI = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😄",
    "sad":      "😢",
    "surprise": "😲",
    "neutral":  "😐",
}


# ======================================================================
# Основной класс
# ======================================================================
class FaceRecognizer:
    """Распознавание лиц (кто) + эмоций (что чувствует)."""

    def __init__(self, threshold: float = 0.45):
        """
        threshold — порог расстояния для идентификации.
        Чем меньше, тем строже (реже узнаёт).
        """
        self.threshold = threshold
        self.known_names = []      # список имён
        self.known_embeddings = [] # список np.array (векторов лиц)
        self._load_db()

    # ------------------------------------------------------------------
    # База лиц
    # ------------------------------------------------------------------
    def _load_db(self):
        """Загружает кэш эмбеддингов лиц."""
        if DB_PATH.exists():
            try:
                with open(DB_PATH, "rb") as f:
                    data = pickle.load(f)
                self.known_names = data.get("names", [])
                self.known_embeddings = [np.array(e) for e in data.get("embeddings", [])]
                print(f"[face] Загружено лиц: {len(self.known_names)}")
            except Exception as e:
                print(f"[face] Ошибка загрузки базы: {e}")

    def _save_db(self):
        with open(DB_PATH, "wb") as f:
            pickle.dump({
                "names": self.known_names,
                "embeddings": [e.tolist() for e in self.known_embeddings],
            }, f)

    def add_face(self, name: str, frame_bgr: np.ndarray) -> bool:
        """
        Добавляет лицо с кадра в базу.
        Возвращает True, если лицо найдено и добавлено.
        """
        DeepFace = _get_deepface()
        try:
            reps = DeepFace.represent(
                img_path=frame_bgr,
                model_name="Facenet",
                enforce_detection=True,
                detector_backend="opencv",
            )
            if not reps:
                return False
            emb = np.array(reps[0]["embedding"], dtype=np.float32)
        except Exception as e:
            print(f"[face] Не удалось получить эмбеддинг: {e}")
            return False

        self.known_names.append(name)
        self.known_embeddings.append(emb)

        # Сохраняем и оригинальный кадр
        person_dir = FACES_DIR / name
        person_dir.mkdir(exist_ok=True)
        idx = len(list(person_dir.glob("*.jpg")))
        cv2.imwrite(str(person_dir / f"{idx:03d}.jpg"), frame_bgr)

        self._save_db()
        print(f"[face] Добавлено лицо: {name}")
        return True

    def rebuild_db(self):
        """Пересобрать базу из папок data/faces/<имя>/*.jpg."""
        DeepFace = _get_deepface()
        self.known_names = []
        self.known_embeddings = []
        for person_dir in sorted(FACES_DIR.iterdir()):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            for img_path in person_dir.glob("*.jpg"):
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    reps = DeepFace.represent(
                        img_path=img,
                        model_name="Facenet",
                        enforce_detection=True,
                        detector_backend="opencv",
                    )
                    if reps:
                        emb = np.array(reps[0]["embedding"], dtype=np.float32)
                        self.known_names.append(name)
                        self.known_embeddings.append(emb)
                except Exception:
                    continue
        self._save_db()
        print(f"[face] База пересобрана: {len(self.known_names)} лиц")

    # ------------------------------------------------------------------
    # Анализ кадра
    # ------------------------------------------------------------------
    def analyze(self, frame_bgr: np.ndarray):
        """
        Возвращает (name, emotion_en, emotion_ru, emoji, face_box) или
                (UNKNOWN_NAME, None, None, None, None) если лицо не найдено.

        name        — имя или "Какой-то чел"
        emotion_en  — 'happy' / 'sad' / ...
        emotion_ru  — "Радость" / "Грусть" / ...
        emoji       — "😄" / "😢" / ...
        face_box    — (x, y, w, h) — координаты лица
        """
        DeepFace = _get_deepface()

        # 1) Ищем лицо и эмоции
        try:
            results = DeepFace.analyze(
                img_path=frame_bgr,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="opencv",
                silent=True,
            )
        except Exception as e:
            print(f"[face] Ошибка анализа: {e}")
            return UNKNOWN_NAME, None, None, None, None

        if not results:
            return UNKNOWN_NAME, None, None, None, None

        # DeepFace иногда возвращает список, иногда dict
        if isinstance(results, list):
            r = results[0]
        else:
            r = results

        # Координаты лица
        region = r.get("region", {})
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)
        face_box = (x, y, w, h) if w > 0 and h > 0 else None

        # Эмоция
        emotion_en = r.get("dominant_emotion", "neutral")
        emotion_ru = EMOTION_RU.get(emotion_en, emotion_en)
        emoji = EMOTION_EMOJI.get(emotion_en, "")

        # 2) Определяем имя
        name = UNKNOWN_NAME
        if self.known_embeddings:
            try:
                reps = DeepFace.represent(
                    img_path=frame_bgr,
                    model_name="Facenet",
                    enforce_detection=False,
                    detector_backend="opencv",
                )
                if reps:
                    cur = np.array(reps[0]["embedding"], dtype=np.float32)
                    # Косинусное расстояние
                    cur_norm = cur / (np.linalg.norm(cur) + 1e-9)
                    best_dist = 1e9
                    best_idx = -1
                    for i, emb in enumerate(self.known_embeddings):
                        e_norm = emb / (np.linalg.norm(emb) + 1e-9)
                        dist = 1.0 - float(np.dot(cur_norm, e_norm))
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = i
                    if best_dist < self.threshold and best_idx >= 0:
                        name = self.known_names[best_idx]
            except Exception as e:
                print(f"[face] Ошибка идентификации: {e}")

        return name, emotion_en, emotion_ru, emoji, face_box