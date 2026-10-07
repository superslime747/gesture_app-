"""
Экран виртуальной мыши.
Слева — видео с камеры и зоной отслеживания, справа — статус,
текущий жест и справка по жестам.
"""
import customtkinter as ctk
from PIL import Image
import numpy as np
import cv2

from virtual_mouse import VirtualMouse
from virtual_mouse import (
    ZONE_X_MIN, ZONE_X_MAX, ZONE_Y_MIN, ZONE_Y_MAX,
)


class MouseView(ctk.CTkFrame):
    """Вид виртуальной мыши."""

    def __init__(self, master, camera, classifier, app):
        super().__init__(master, fg_color="transparent")
        self.camera = camera
        self.classifier = classifier
        self.app = app

        # Виртуальная мышь
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        self.mouse = VirtualMouse(
            screen_w, screen_h,
            sensitivity=app.settings.get("cursor_sensitivity", 5),
            smoothing=app.settings.get("smoothing", 5),
        )

        # Левая колонка — видео (растягивается)
        self.grid_columnconfigure(0, weight=3, minsize=500)
        # Правая колонка — панель (фиксированная)
        self.grid_columnconfigure(1, weight=0, minsize=340)
        self.grid_rowconfigure(0, weight=1)

        # ---------- Левая часть: видео ----------
        left = ctk.CTkFrame(self, corner_radius=16)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        top_left = ctk.CTkFrame(left, fg_color="transparent")
        top_left.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        back_btn = ctk.CTkButton(top_left, text="← Назад", width=100,
                                 command=app.show_mode_selection)
        back_btn.pack(side="left")

        self.video_label = ctk.CTkLabel(left, text="Загрузка камеры...")
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # ---------- Правая часть ----------
        right = ctk.CTkFrame(self, corner_radius=16, width=340)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=0)
        right.grid_columnconfigure(0, weight=1)
        right.grid_propagate(False)

        # Статус
        ctk.CTkLabel(right, text="Статус:", font=ctk.CTkFont(size=14)).grid(
            row=0, column=0, padx=20, pady=(20, 0), sticky="w")
        self.status_label = ctk.CTkLabel(
            right, text="Ожидание", font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#4A9EFF", anchor="w", justify="left")
        self.status_label.grid(row=1, column=0, padx=20, pady=(0, 16), sticky="ew")

        # Текущий жест
        ctk.CTkLabel(right, text="Текущий жест:", font=ctk.CTkFont(size=14)).grid(
            row=2, column=0, padx=20, pady=(0, 0), sticky="w")
        self.gesture_label = ctk.CTkLabel(
            right, text="—", font=ctk.CTkFont(size=22, weight="bold"),
            anchor="w", justify="left")
        self.gesture_label.grid(row=3, column=0, padx=20, pady=(0, 16), sticky="ew")

        # Разделитель
        sep = ctk.CTkFrame(right, height=1, fg_color="#444444")
        sep.grid(row=4, column=0, padx=20, pady=8, sticky="ew")

        # Справка по жестам
        ctk.CTkLabel(right, text="Управление жестами:",
                     font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=5, column=0, padx=20, pady=(4, 8), sticky="w")

        hints = [
            ("☝  Указательный", "движение"),
            ("👌  OK", "левый клик"),
            ("✌  Два пальца", "правый клик"),
            ("✊  Кулак", "захват (drag)"),
            ("✋  Открытая ладонь", "пауза"),
            ("👍  Большой палец", "скролл ↑/↓"),
        ]
        for i, (gest, action) in enumerate(hints):
            row_frame = ctk.CTkFrame(right, fg_color="transparent")
            row_frame.grid(row=6 + i, column=0, padx=20, pady=2, sticky="ew")
            row_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                row_frame, text=gest, font=ctk.CTkFont(size=13),
                anchor="w"
            ).grid(row=0, column=0, sticky="w")

            ctk.CTkLabel(
                row_frame, text=action, font=ctk.CTkFont(size=12),
                text_color="#AAAAAA", anchor="e"
            ).grid(row=0, column=1, sticky="e")

        # Запускаем цикл
        self._update_loop()

    # ------------------------------------------------------------------
    def _update_loop(self):
        if not self.winfo_exists():
            return

        frame, landmarks = self.camera.get_frame()
        gesture_label = None
        gesture_name = "—"

        if landmarks is not None and self.classifier.is_ready:
            gesture_label, gesture_name, conf = self.classifier.predict(landmarks)
            if gesture_label is None:
                gesture_name = "—"

        # Управление мышью
        try:
            self.mouse.update(landmarks, gesture_label)
        except Exception as e:
            print(f"[mouse] Ошибка управления: {e}")

        # Рисуем зону отслеживания
        if frame is not None:
            h, w = frame.shape[:2]
            x1 = int(ZONE_X_MIN * w)
            x2 = int(ZONE_X_MAX * w)
            y1 = int(ZONE_Y_MIN * h)
            y2 = int(ZONE_Y_MAX * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (74, 158, 255), 2)

            rgb = frame[:, :, ::-1]
            img = Image.fromarray(rgb)
            lw = max(self.video_label.winfo_width(), 320)
            lh = max(self.video_label.winfo_height(), 240)
            img = img.resize((lw, lh), Image.BILINEAR)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(lw, lh))
            self.video_label.configure(image=ctk_img, text="")
            self.video_label._image = ctk_img

        # Обновляем статус и текущий жест
        self.status_label.configure(text=self.mouse.last_action)
        self.gesture_label.configure(text=gesture_name)

        self.after(30, self._update_loop)

    # ------------------------------------------------------------------
    def apply_settings(self, settings: dict):
        self.mouse.set_sensitivity(settings.get("cursor_sensitivity", 5))
        self.mouse.set_smoothing(settings.get("smoothing", 5))

    def on_hide(self):
        try:
            self.mouse.stop()
        except Exception:
            pass