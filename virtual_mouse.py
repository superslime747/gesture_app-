"""
Логика виртуальной мыши.

Жесты:
  ☝  Указательный      → движение курсора
  👌  OK               → левый клик
  ✌  Два пальца        → правый клик
  ✊  Кулак             → drag
  ✋  Открытая ладонь   → пауза
  👍  Большой палец     → скролл (направление — по геометрии руки)

Особенности:
- Плавное движение курсора (адаптивное EMA + интерполяция).
- Скролл с плавным разгоном, пока жест удерживается.
- Скролл на Windows через ctypes.SendInput (надёжнее pyautogui).
"""
import time
import platform
import ctypes
import numpy as np
import pyautogui

pyautogui.FAILSAFE = False

# ----------------------------------------------------------------------
# Метки жестов (совпадают с классификатором)
# ----------------------------------------------------------------------
GESTURE_ROCK = 0          # кулак
GESTURE_SCISSORS = 1      # ножницы (два пальца)
GESTURE_PAPER = 2         # открытая ладонь
GESTURE_INDEX = 3         # указательный палец
GESTURE_PINCH = 4         # щипок (не используется)
GESTURE_THUMB = 5         # большой палец (вверх/вниз — по геометрии)
GESTURE_OK = 6            # OK → левый клик
GESTURE_THUMB_DOWN = 7    # если когда-то добавите отдельный класс

# ----------------------------------------------------------------------
# Зона отслеживания (доли от размера кадра)
# ----------------------------------------------------------------------
ZONE_X_MIN, ZONE_X_MAX = 0.2, 0.8
ZONE_Y_MIN, ZONE_Y_MAX = 0.2, 0.8

# ----------------------------------------------------------------------
# Тайминги кликов
# ----------------------------------------------------------------------
CLICK_HOLD_SEC = 0.25         # OK держится столько для клика
CLICK_COOLDOWN_SEC = 0.5      # пауза между кликами

# ----------------------------------------------------------------------
# Параметры скролла
# ----------------------------------------------------------------------
SCROLL_BASE_SPEED = 12        # "щелчков" в секунду в начале жеста
SCROLL_MAX_SPEED = 60         # максимум после разгона
SCROLL_ACCEL_TIME = 0.8       # за сколько секунд разгоняется до максимума
SCROLL_MAX_RATE_HZ = 60       # не чаще 60 раз в секунду

# ----------------------------------------------------------------------
# Интерполяция курсора
# ----------------------------------------------------------------------
INTERPOLATION_STEPS = 3


# ======================================================================
# Низкоуровневый скролл через SendInput (Windows)
# ======================================================================
_IS_WINDOWS = platform.system() == "Windows"

if _IS_WINDOWS:
    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.c_ulong),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class _INPUT(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_ulong),
            ("mi", _MOUSEINPUT),
        ]

    _INPUT_MOUSE = 0
    _MOUSEEVENTF_WHEEL = 0x0800
    _WHEEL_DELTA = 120


def _scroll_windows(clicks: int):
    """Отправляет событие колеса мыши. clicks: '+' вверх, '-' вниз."""
    if not _IS_WINDOWS:
        pyautogui.scroll(clicks)
        return

    amount = _WHEEL_DELTA * clicks
    mouse_input = _MOUSEINPUT(
        dx=0, dy=0,
        mouseData=ctypes.c_ulong(amount & 0xFFFFFFFF),
        dwFlags=_MOUSEEVENTF_WHEEL,
        time=0,
        dwExtraInfo=None,
    )
    inp = _INPUT(type=_INPUT_MOUSE, mi=mouse_input)
    ctypes.windll.user32.SendInput(
        1, ctypes.byref(inp), ctypes.sizeof(inp)
    )


# ======================================================================
# Класс виртуальной мыши
# ======================================================================
class VirtualMouse:
    """Управление курсором жестами."""

    def __init__(self, screen_w: int, screen_h: int,
                 sensitivity: int = 5, smoothing: int = 5):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.sensitivity = sensitivity
        self.smoothing = smoothing

        # Позиция курсора (EMA)
        self._prev_x = None
        self._prev_y = None

        # FPS
        self._last_update_time = None
        self._current_fps = 30.0

        # Клик (OK)
        self._click_start = 0.0
        self._last_click_time = 0.0

        # Drag
        self._dragging = False

        # Скролл
        self._scroll_start_time = None
        self._scroll_direction = 0
        self._last_scroll_time = 0.0
        self._scroll_accum = 0.0

        self.last_action = "Ожидание"

    # ------------------------------------------------------------------
    def set_sensitivity(self, value: int):
        self.sensitivity = max(1, min(10, int(value)))

    def set_smoothing(self, value: int):
        self.smoothing = max(0, min(10, int(value)))

    # ------------------------------------------------------------------
    def update(self, landmarks, gesture_label):
        if landmarks is None:
            self.last_action = "Рука не найдена"
            self._reset_scroll()
            self._click_start = 0.0
            return

        now = time.time()
        if self._last_update_time is not None:
            dt = now - self._last_update_time
            if dt > 0:
                inst_fps = 1.0 / dt
                self._current_fps = 0.9 * self._current_fps + 0.1 * inst_fps
        self._last_update_time = now

        target_x, target_y = self._landmarks_to_screen(landmarks)

        # --- Приоритет: скролл (большой палец) ---
        if gesture_label in (GESTURE_THUMB, GESTURE_THUMB_DOWN):
            self._handle_scroll(landmarks, gesture_label, now)
            return
        else:
            self._reset_scroll()

        # --- Пауза ---
        if gesture_label == GESTURE_PAPER:
            self.last_action = "Пауза"
            self._click_start = 0.0
            return

        # --- Drag (кулак) ---
        if gesture_label == GESTURE_ROCK:
            if not self._dragging:
                pyautogui.mouseDown()
                self._dragging = True
            self.last_action = "Захват (drag)"
            self._move_cursor(target_x, target_y)
            self._click_start = 0.0
            return
        else:
            if self._dragging:
                pyautogui.mouseUp()
                self._dragging = False

        # --- Левый клик (OK) ---
        if gesture_label == GESTURE_OK:
            if self._click_start == 0.0:
                self._click_start = now
            elif (now - self._click_start >= CLICK_HOLD_SEC
                  and now - self._last_click_time >= CLICK_COOLDOWN_SEC):
                pyautogui.click()
                self._last_click_time = now
                self._click_start = 0.0
            self.last_action = "Клик"
            return
        else:
            self._click_start = 0.0

        # --- Правый клик (два пальца) ---
        if gesture_label == GESTURE_SCISSORS:
            if now - self._last_click_time >= CLICK_COOLDOWN_SEC:
                pyautogui.rightClick()
                self._last_click_time = now
            self.last_action = "Правый клик"
            return

        # --- Движение (указательный) ---
        if gesture_label == GESTURE_INDEX:
            self._move_cursor(target_x, target_y)
            self.last_action = "Движение"
            return

        self.last_action = "Ожидание"

    # ------------------------------------------------------------------
    def _landmarks_to_screen(self, landmarks):
        x, y = landmarks[8][0], landmarks[8][1]
        nx = (x - ZONE_X_MIN) / (ZONE_X_MAX - ZONE_X_MIN)
        ny = (y - ZONE_Y_MIN) / (ZONE_Y_MAX - ZONE_Y_MIN)
        nx = float(np.clip(nx, 0.0, 1.0))
        ny = float(np.clip(ny, 0.0, 1.0))
        return int(nx * self.screen_w), int(ny * self.screen_h)

    # ------------------------------------------------------------------
    def _reset_scroll(self):
        self._scroll_start_time = None
        self._scroll_direction = 0
        self._scroll_accum = 0.0

    def _handle_scroll(self, landmarks, gesture_label, now):
        """Определяет направление и делает скролл."""
        if gesture_label == GESTURE_THUMB_DOWN:
            direction = -1
        else:
            thumb_tip_y = landmarks[4][1]
            base_y = landmarks[5][1]
            direction = 1 if thumb_tip_y < base_y else -1

        if self._scroll_start_time is None or direction != self._scroll_direction:
            self._scroll_start_time = now
            self._scroll_direction = direction
            self._scroll_accum = 0.0
            self._last_scroll_time = now

        held = now - self._scroll_start_time
        t = min(1.0, held / SCROLL_ACCEL_TIME)
        speed = SCROLL_BASE_SPEED + (SCROLL_MAX_SPEED - SCROLL_BASE_SPEED) * t

        dt = now - self._last_scroll_time
        if dt < 1.0 / SCROLL_MAX_RATE_HZ:
            return
        self._last_scroll_time = now

        amount = speed * dt + self._scroll_accum
        clicks = int(amount)
        self._scroll_accum = amount - clicks

        if clicks > 0:
            _scroll_windows(clicks * direction)

        self.last_action = "Скролл ↑" if direction > 0 else "Скролл ↓"

    # ------------------------------------------------------------------
    def _move_cursor(self, target_x: int, target_y: int):
        sens = 0.6 + (self.sensitivity - 1) * (1.0 / 9.0)
        base_alpha = 1.0 - (self.smoothing / 10.0) * 0.85
        fps_factor = max(0.5, min(1.5, self._current_fps / 30.0))
        alpha = min(1.0, base_alpha * fps_factor)

        if self._prev_x is None or self._prev_y is None:
            self._prev_x = float(target_x)
            self._prev_y = float(target_y)
            start_x, start_y = self._prev_x, self._prev_y
        else:
            start_x = self._prev_x
            start_y = self._prev_y
            self._prev_x += alpha * (target_x - self._prev_x) * sens
            self._prev_y += alpha * (target_y - self._prev_y) * sens

        new_x = self._prev_x
        new_y = self._prev_y

        dx = new_x - start_x
        dy = new_y - start_y
        dist = (dx * dx + dy * dy) ** 0.5

        if dist < 5:
            steps = 1
        elif dist < 40:
            steps = 2
        else:
            steps = INTERPOLATION_STEPS

        for i in range(1, steps + 1):
            t = i / steps
            ix = int(start_x + dx * t)
            iy = int(start_y + dy * t)
            pyautogui.moveTo(ix, iy, duration=0)

    # ------------------------------------------------------------------
    def stop(self):
        if self._dragging:
            pyautogui.mouseUp()
            self._dragging = False