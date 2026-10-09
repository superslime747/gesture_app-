"""
Скрипт сбора данных для обучения классификатора жестов и лиц.

- Жесты: клавиши 0..6, SPACE — запись 100 кадров landmarks
- Лица:  клавиша F — сохранить лицо (откроется окно ввода имени)
- Q — выход

Русский текст рисуется через Pillow.
"""
# ---- Глушим логи TensorFlow ДО импорта ----
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# --------------------------------------------

import cv2
import csv
import time
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from camera import CameraStream, normalize_landmarks

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "gestures.csv"

GESTURES = {
    0: "Камень (кулак)",
    1: "Ножницы (два пальца)",
    2: "Бумага (открытая ладонь)",
    3: "Указательный палец",
    4: "Щипок",
    5: "Большой палец вверх",
    6: "OK",
}

FRAMES_PER_CAPTURE = 100


# ------------------------------------------------------------------
# Шрифт
# ------------------------------------------------------------------
def _load_font(size: int):
    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


FONT_BIG = _load_font(22)
FONT_MED = _load_font(18)
FONT_SMALL = _load_font(14)


def draw_cyrillic(frame_bgr, lines):
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for text, x, y, color_bgr, font in lines:
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------------
# CSV
# ------------------------------------------------------------------
def ensure_csv_header():
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = [f"f{i}" for i in range(63)] + ["label"]
            writer.writerow(header)


def save_samples(features_list, label):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for feat in features_list:
            writer.writerow(list(feat) + [label])


# ------------------------------------------------------------------
# Модальное окно ввода имени (через Tkinter)
# ------------------------------------------------------------------
def ask_name_dialog():
    """Открывает маленькое окно для ввода имени. Возвращает строку или None."""
    import tkinter as tk
    from tkinter import simpledialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    name = simpledialog.askstring("Сохранение лица",
                                  "Введите имя человека:",
                                  parent=root)
    root.destroy()
    return name.strip() if name else None


# ------------------------------------------------------------------
# Сохранение лица в фоне
# ------------------------------------------------------------------
def save_face_async(frame_bgr, recognizer, on_done):
    """
    Сохраняет лицо в отдельном потоке.
    on_done(success: bool, message: str) — вызывается по завершении.
    """
    def worker():
        try:
            name = ask_name_dialog()
            if not name:
                on_done(False, "Отменено")
                return
            ok = recognizer.add_face(name, frame_bgr)
            if ok:
                on_done(True, f"Лицо '{name}' сохранено")
            else:
                on_done(False, "Лицо не найдено в кадре")
        except Exception as e:
            on_done(False, f"Ошибка: {e}")

    threading.Thread(target=worker, daemon=True).start()


# ------------------------------------------------------------------
# Основной цикл
# ------------------------------------------------------------------
def main():
    ensure_csv_header()

    # Статус для вывода в кадре
    status_msg = "Загрузка моделей распознавания лиц..."
    status_until = time.time() + 60  # держим сообщение до готовности
    face_status = {"ready": False, "msg": status_msg}

    # Загружаем FaceRecognizer в фоне заранее
    face_recognizer = {"obj": None}

    def load_recognizer():
        try:
            from face_recognizer import FaceRecognizer
            face_recognizer["obj"] = FaceRecognizer()
            face_status["ready"] = True
            face_status["msg"] = "Модели лиц готовы (F — сохранить лицо)"
        except Exception as e:
            face_status["msg"] = f"Ошибка моделей лиц: {e}"

    threading.Thread(target=load_recognizer, daemon=True).start()

    cam = CameraStream(camera_index=0)
    cam.start()

    print("=" * 60)
    print("СБОР ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
    print("=" * 60)
    print("Управление:")
    print("  0..6  — выбрать жест")
    print("  SPACE — записать 100 кадров жеста")
    print("  F     — сохранить лицо (для распознавания)")
    print("  Q     — выход")
    print("=" * 60)
    print("Загрузка моделей распознавания лиц... (несколько секунд)")

    current_gesture = 0
    last_face_msg = ""
    last_face_msg_time = 0.0

    def show_face_result(success, message):
        nonlocal last_face_msg, last_face_msg_time
        last_face_msg = ("✓ " if success else "✗ ") + message
        last_face_msg_time = time.time()
        print(last_face_msg)

    while True:
        frame, landmarks = cam.get_frame()
        if frame is None:
            continue

        gesture_name = GESTURES[current_gesture]
        hand_status = "Рука: OK" if landmarks is not None else "Рука: не найдена"
        hand_color = (0, 255, 0) if landmarks is not None else (0, 0, 255)

        # Собираем строки для вывода
        lines = [
            (f"Жест: {gesture_name}", 10, 55, (74, 158, 255), FONT_BIG),
            (hand_status, 10, 85, hand_color, FONT_MED),
        ]

        # Статус моделей лиц (пока не готовы или свежее сообщение)
        now = time.time()
        if not face_status["ready"]:
            lines.append((face_status["msg"], 10, 115, (255, 200, 0), FONT_SMALL))
        elif now - last_face_msg_time < 3.0 and last_face_msg:
            lines.append((last_face_msg, 10, 115, (0, 255, 200), FONT_SMALL))

        # Подсказка внизу
        lines.append(("SPACE — записать | 0-6 — жест | F — лицо | Q — выход",
                      10, frame.shape[0] - 30, (200, 200, 200), FONT_SMALL))

        frame = draw_cyrillic(frame, lines)

        cv2.imshow("Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break

        elif ord('0') <= key <= ord('6'):
            current_gesture = key - ord('0')
            print(f"Выбран жест: {GESTURES[current_gesture]}")

        elif key == ord(' '):
            if landmarks is None:
                print("Рука не найдена — нечего сохранять")
                continue
            print(f"Запись {FRAMES_PER_CAPTURE} кадров для '{gesture_name}'...")
            samples = []
            start = time.time()
            while len(samples) < FRAMES_PER_CAPTURE and time.time() - start < 5.0:
                f2, lm2 = cam.get_frame()
                if lm2 is not None:
                    feat = normalize_landmarks(lm2)
                    if feat is not None:
                        samples.append(feat)
                if f2 is not None:
                    f2 = draw_cyrillic(f2, [
                        (f"Запись: {len(samples)}/{FRAMES_PER_CAPTURE}",
                         10, 115, (0, 255, 255), FONT_BIG),
                    ])
                    cv2.imshow("Data Collector", f2)
                    cv2.waitKey(1)
                time.sleep(0.01)

            save_samples(samples, current_gesture)
            print(f"Сохранено {len(samples)} примеров для '{gesture_name}'")

        elif key == ord('f') or key == ord('F'):
            if not face_status["ready"]:
                print("Модели лиц ещё грузятся, подождите...")
                continue

            if face_recognizer["obj"] is None:
                print("FaceRecognizer не загружен")
                continue

            # Запускаем сохранение в фоне — окно OpenCV не блокируется
            save_face_async(frame.copy(),
                            face_recognizer["obj"],
                            show_face_result)

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()