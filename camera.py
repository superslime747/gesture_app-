"""
Модуль работы с камерой и MediaPipe Hands.
Предоставляет класс CameraStream, который в отдельном потоке читает кадры,
прогоняет их через MediaPipe и складывает результат в буфер.
"""
import cv2
import mediapipe as mp
import numpy as np
import threading
import time


class CameraStream:
    """
    Потокобезопасный поток с камеры с распознаванием landmarks руки.
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height

        # MediaPipe Hands — используем быструю модель (model_complexity=0)
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,          # быстрее
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.cap = None
        self.running = False
        self.thread = None

        # Буфер с текущим кадром и landmarks
        self._lock = threading.Lock()
        self._frame = None                # BGR-кадр с нарисованными landmarks
        self._raw_frame = None            # сырой BGR-кадр (без разметки)
        self._landmarks = None            # np.array (21,3) или None
        self._fps = 0.0

    # ------------------------------------------------------------------
    # Запуск/остановка
    # ------------------------------------------------------------------
    def start(self):
        """Запускает поток чтения камеры."""
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if _is_windows() else cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть камеру с индексом {self.camera_index}")

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Останавливает поток и освобождает ресурсы."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        try:
            self.hands.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Основной цикл
    # ------------------------------------------------------------------
    def _loop(self):
        prev_time = time.time()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            # Отзеркаливаем кадр (эффект зеркала — интуитивно удобнее)
            frame = cv2.flip(frame, 1)

            # MediaPipe требует RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            landmarks = None
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                # Рисуем landmarks и соединения
                self.mp_draw.draw_landmarks(
                    frame, hand, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(74, 158, 255), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1),
                )
                # Извлекаем массив (21,3)
                landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand.landmark],
                    dtype=np.float32,
                )

            # FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps = 1.0 / dt if dt > 0 else 0.0

            # Рисуем FPS в углу
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            with self._lock:
                self._frame = frame
                self._raw_frame = frame.copy()
                self._landmarks = landmarks
                self._fps = fps

    # ------------------------------------------------------------------
    # Получение данных
    # ------------------------------------------------------------------
    def get_frame(self):
        """Возвращает текущий кадр (BGR) и landmarks (np.array (21,3) или None)."""
        with self._lock:
            if self._frame is None:
                return None, None
            return self._frame.copy(), (self._landmarks.copy() if self._landmarks is not None else None)

    def get_fps(self) -> float:
        with self._lock:
            return self._fps


# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------
def _is_windows() -> bool:
    import platform
    return platform.system() == "Windows"


def list_available_cameras(max_index: int = 5):
    """Возвращает список доступных индексов камер."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if _is_windows() else cv2.CAP_ANY)
        if cap.isOpened():
            available.append(i)
        cap.release()
    return available


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Нормализует landmarks относительно запястья (landmark #0)
    и масштабирует по максимальному расстоянию, чтобы жест
    не зависел от положения и размера руки в кадре.
    Возвращает плоский вектор длиной 63 (21 * 3).
    """
    if landmarks is None:
        return None
    # Смещаем относительно запястья
    wrist = landmarks[0]
    shifted = landmarks - wrist
    # Масштаб — максимальное расстояние от запястья
    scale = np.linalg.norm(shifted[:, :2], axis=1).max()
    if scale > 1e-6:
        shifted = shifted / scale
    return shifted.flatten().astype(np.float32)