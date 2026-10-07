"""
Логика виртуальной мыши.
Преобразует landmarks руки в движения курсора и клики через pyautogui.
"""
import time
import numpy as np
import pyautogui

# Отключаем "уголок" pyautogui — иначе при наведении в угол срабатывает защита
pyautogui.FAILSAFE = False

# Жесты (совпадают с классификатором)
GESTURE_INDEX = 3      # Указательный — движение
GESTURE_PINCH = 4      # Щипок — левый клик
GESTURE_ROCK = 0       # Кулак — захват/drag
GESTURE_PAPER = 2      # Открытая ладонь — пауза
# Для правого клика используем "Ножницы" (2 пальца) — метка 1
GESTURE_SCISSORS = 1

# Зона отслеживания в кадре (доли от ширины/высоты)
ZONE_X_MIN, ZONE_X_MAX = 0.2, 0.8
ZONE_Y_MIN, ZONE_Y_MAX = 0.2, 0.8

# Тайминги
PINCH_HOLD_SEC = 0.3       # щипок должен удерживаться столько для клика
CLICK_COOLDOWN_SEC = 0.5   # пауза между кликами


class VirtualMouse:
    """Управление курсором жестами."""

    def __init__(self, screen_w: int, screen_h: int,
                 sensitivity: int = 5, smoothing: int = 5):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.sensitivity = sensitivity       # 1..10
        self.smoothing = smoothing           # 0..10

        # Предыдущая позиция курсора (для сглаживания)
        self._prev_x = None
        self._prev_y = None

        # Состояние щипка
        self._pinch_start = 0.0
        self._last_click_time = 0.0

        # Состояние drag (кулак)
        self._dragging = False

        self.last_action = "Ожидание"

    # ------------------------------------------------------------------
    # Обновление настроек
    # ------------------------------------------------------------------
    def set_sensitivity(self, value: int):
        self.sensitivity = max(1, min(10, int(value)))

    def set_smoothing(self, value: int):
        self.smoothing = max(0, min(10, int(value)))

    # ------------------------------------------------------------------
    # Основная логика
    # ------------------------------------------------------------------
    def update(self, landmarks, gesture_label):
        """
        landmarks — np.array (21,3) или None.
        gesture_label — метка жеста или None.
        """
        if landmarks is None:
            self.last_action = "Рука не найдена"
            return

        # Кончик указательного пальца — landmark #8
        x, y = landmarks[8][0], landmarks[8][1]

        # Ограничиваем зону отслеживания
        nx = (x - ZONE_X_MIN) / (ZONE_X_MAX - ZONE_X_MIN)
        ny = (y - ZONE_Y_MIN) / (ZONE_Y_MAX - ZONE_Y_MIN)
        nx = float(np.clip(nx, 0.0, 1.0))
        ny = float(np.clip(ny, 0.0, 1.0))

        # Маппинг на экран
        target_x = int(nx * self.screen_w)
        target_y = int(ny * self.screen_h)

        # Действие по жесту
        now = time.time()

        if gesture_label == GESTURE_PAPER:
            # Открытая ладонь — пауза
            self.last_action = "Пауза"
            return

        if gesture_label == GESTURE_ROCK:
            # Кулак — захват/перетаскивание
            if not self._dragging:
                pyautogui.mouseDown()
                self._dragging = True
            self.last_action = "Захват (drag)"
            self._move_cursor(target_x, target_y)
            return
        else:
            # Отпускаем drag, если был
            if self._dragging:
                pyautogui.mouseUp()
                self._dragging = False

        if gesture_label == GESTURE_PINCH:
            # Щипок — левый клик (удержание 0.3 сек)
            if self._pinch_start == 0.0:
                self._pinch_start = now
            elif (now - self._pinch_start >= PINCH_HOLD_SEC
                  and now - self._last_click_time >= CLICK_COOLDOWN_SEC):
                pyautogui.click()
                self._last_click_time = now
                self._pinch_start = 0.0
            self.last_action = "Клик"
            return
        else:
            self._pinch_start = 0.0

        if gesture_label == GESTURE_SCISSORS:
            # Два пальца — правый клик
            if now - self._last_click_time >= CLICK_COOLDOWN_SEC:
                pyautogui.rightClick()
                self._last_click_time = now
            self.last_action = "Правый клик"
            return

        if gesture_label == GESTURE_INDEX:
            # Указательный — движение курсора
            self._move_cursor(target_x, target_y)
            self.last_action = "Движение"
            return

        # Неизвестный жест — ничего не делаем
        self.last_action = "Ожидание"

    # ------------------------------------------------------------------
    # Перемещение курсора со сглаживанием
    # ------------------------------------------------------------------
    def _move_cursor(self, target_x: int, target_y: int):
        # Чувствительность: множитель отклонения от центра (1..10 → 0.5..1.5)
        sens = 0.5 + (self.sensitivity - 1) * (1.0 / 9.0) * 1.0

        if self._prev_x is None or self._prev_y is None:
            self._prev_x, self._prev_y = target_x, target_y
        else:
            # Сглаживание (EMA): alpha тем меньше, чем больше smoothing
            # smoothing=0 → alpha=1.0 (без сглаживания)
            # smoothing=10 → alpha=0.1 (сильное сглаживание)
            alpha = 1.0 - (self.smoothing / 10.0) * 0.9
            self._prev_x = self._prev_x + alpha * (target_x - self._prev_x) * sens
            self._prev_y = self._prev_y + alpha * (target_y - self._prev_y) * sens

        pyautogui.moveTo(int(self._prev_x), int(self._prev_y), duration=0)

    def stop(self):
        """Отпускаем мышь, если был drag."""
        if self._dragging:
            pyautogui.mouseUp()
            self._dragging = False