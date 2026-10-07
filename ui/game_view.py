"""
Экран игры "Камень-Ножницы-Бумага".
Слева — видео с камеры и landmarks, справа — состояние игры.
"""
import time
import customtkinter as ctk
from PIL import Image
import numpy as np

from game_rps import RPSGame, RPS_LABELS


class GameView(ctk.CTkFrame):
    """Вид игры КНБ."""

    def __init__(self, master, camera, classifier, app):
        super().__init__(master, fg_color="transparent")
        self.camera = camera
        self.classifier = classifier
        self.app = app

        self.game = RPSGame()
        self._last_update = 0.0

        # Левая колонка (видео) — растягивается
        self.grid_columnconfigure(0, weight=3, minsize=500)
        # Правая колонка (панель) — фиксированная ширина
        self.grid_columnconfigure(1, weight=0, minsize=340)
        self.grid_rowconfigure(0, weight=1)

        # ---------- Левая часть: видео ----------
        left = ctk.CTkFrame(self, corner_radius=16)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        top_left = ctk.CTkFrame(left, fg_color="transparent")
        top_left.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        top_left.grid_columnconfigure(1, weight=1)

        back_btn = ctk.CTkButton(top_left, text="← Назад", width=100,
                                 command=app.show_mode_selection)
        back_btn.grid(row=0, column=0, sticky="w")

        self.score_label = ctk.CTkLabel(
            top_left, text="Игрок 0 : 0 ИИ",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.score_label.grid(row=0, column=1, sticky="e")

        self.video_label = ctk.CTkLabel(left, text="Загрузка камеры...")
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # ---------- Правая часть ----------
        right = ctk.CTkFrame(self, corner_radius=16, width=340)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=0)
        right.grid_columnconfigure(0, weight=1)
        right.grid_propagate(False)  # фиксируем ширину

        # Жест ИИ
        ctk.CTkLabel(right, text="Жест ИИ:", font=ctk.CTkFont(size=14)).grid(
            row=0, column=0, padx=20, pady=(20, 0), sticky="w")
        self.ai_move_label = ctk.CTkLabel(
            right, text="—", font=ctk.CTkFont(size=32, weight="bold"))
        self.ai_move_label.grid(row=1, column=0, padx=20, pady=(0, 12), sticky="w")

        # Жест игрока
        ctk.CTkLabel(right, text="Жест игрока:", font=ctk.CTkFont(size=14)).grid(
            row=2, column=0, padx=20, pady=(8, 0), sticky="w")
        self.player_move_label = ctk.CTkLabel(
            right, text="—", font=ctk.CTkFont(size=32, weight="bold"),
            text_color="#4A9EFF")
        self.player_move_label.grid(row=3, column=0, padx=20, pady=(0, 12), sticky="w")

        # Результат
        ctk.CTkLabel(right, text="Результат:", font=ctk.CTkFont(size=14)).grid(
            row=4, column=0, padx=20, pady=(8, 0), sticky="w")
        self.result_label = ctk.CTkLabel(
            right, text="", font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#4A9EFF")
        self.result_label.grid(row=5, column=0, padx=20, pady=(0, 12), sticky="w")

        # Крупный отсчёт
        self.countdown_label = ctk.CTkLabel(
            right, text="", font=ctk.CTkFont(size=56, weight="bold"),
            text_color="#4A9EFF")
        self.countdown_label.grid(row=6, column=0, padx=20, pady=8, sticky="w")

        # Кнопки
        btn_frame = ctk.CTkFrame(right, fg_color="transparent")
        btn_frame.grid(row=7, column=0, padx=20, pady=(10, 20), sticky="ew")
        btn_frame.grid_columnconfigure(0, weight=1)

        self.start_btn = ctk.CTkButton(
            btn_frame, text="Начать раунд", height=42,
            command=self._start_round)
        self.start_btn.grid(row=0, column=0, pady=5, sticky="ew")

        self.finish_btn = ctk.CTkButton(
            btn_frame, text="Завершить игру", height=42,
            fg_color="#555555", hover_color="#444444",
            command=self._finish)
        self.finish_btn.grid(row=1, column=0, pady=5, sticky="ew")

        # Запускаем цикл обновления
        self._update_loop()

    # ------------------------------------------------------------------
    def _start_round(self):
        self.game.start_round()
        self.result_label.configure(text="")
        self.ai_move_label.configure(text="—")
        self.player_move_label.configure(text="—")
        self.start_btn.configure(state="disabled")

    def _finish(self):
        self.app.show_mode_selection()

    # ------------------------------------------------------------------
    def _update_loop(self):
        """Периодически обновляет кадр, распознаёт жест и логику игры."""
        if not self.winfo_exists():
            return

        frame, landmarks = self.camera.get_frame()
        gesture_label = None
        gesture_name = "Неизвестно"

        if landmarks is not None and self.classifier.is_ready:
            gesture_label, gesture_name, conf = self.classifier.predict(landmarks)

        # Обновляем видео
        if frame is not None:
            rgb = frame[:, :, ::-1]
            img = Image.fromarray(rgb)
            w = max(self.video_label.winfo_width(), 320)
            h = max(self.video_label.winfo_height(), 240)
            img = img.resize((w, h), Image.BILINEAR)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
            self.video_label.configure(image=ctk_img, text="")
            self.video_label._image = ctk_img

        # Обновляем текущий распознанный жест игрока (до завершения раунда)
        if gesture_label is not None and self.game.phase != "result":
            self.player_move_label.configure(text=gesture_name)

        # Логика игры
        now = time.time()
        if now - self._last_update > 0.05:
            self._last_update = now
            finished = self.game.update(gesture_label)
            if finished:
                self._on_round_finished()

        # Обновляем счёт и отсчёт
        self.score_label.configure(
            text=f"Игрок {self.game.player_score} : {self.game.ai_score} ИИ")
        self.countdown_label.configure(text=self.game.countdown_text)

        self.after(30, self._update_loop)

    # ------------------------------------------------------------------
    def _on_round_finished(self):
        if self.game.player_move is not None:
            self.player_move_label.configure(
                text=RPS_LABELS.get(self.game.player_move, "—"))
        if self.game.ai_move is not None:
            self.ai_move_label.configure(
                text=RPS_LABELS.get(self.game.ai_move, "—"))

        self.result_label.configure(text=self.game.result_text)
        self.start_btn.configure(state="normal")
        self.countdown_label.configure(text="")

    def apply_settings(self, settings: dict):
        pass

    def on_hide(self):
        pass