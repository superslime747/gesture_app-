"""
Экран распознавания лица и эмоций.
Русский текст рисуется через Pillow (OpenCV не поддерживает кириллицу).
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import threading

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

from face_recognizer import FaceRecognizer, UNKNOWN_NAME


# ------------------------------------------------------------------
# Шрифт с кириллицей
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


FONT_LABEL = _load_font(22)


def draw_label(frame_bgr, text, x, y, color_bgr=(74, 158, 255)):
    """Рисует русский текст поверх BGR-кадра через Pillow."""
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    # Тень для читаемости на любом фоне
    draw.text((x + 1, y + 1), text, font=FONT_LABEL, fill=(0, 0, 0))
    draw.text((x, y), text, font=FONT_LABEL, fill=color_rgb)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


class FaceView(ctk.CTkFrame):
    """Вид распознавания лица и эмоций."""

    def __init__(self, master, camera, classifier, app):
        super().__init__(master, fg_color="transparent")
        self.camera = camera
        self.app = app

        self.recognizer = None
        self._recognizer_ready = False
        self._analyzing = False

        self._last_name = UNKNOWN_NAME
        self._last_emotion_ru = "—"
        self._last_emoji = ""
        self._last_box = None

        # Левая колонка — видео
        self.grid_columnconfigure(0, weight=3, minsize=500)
        # Правая колонка — панель
        self.grid_columnconfigure(1, weight=0, minsize=340)
        self.grid_rowconfigure(0, weight=1)

        # ---------- Левая часть ----------
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

        ctk.CTkLabel(right, text="Человек:", font=ctk.CTkFont(size=14)).grid(
            row=0, column=0, padx=20, pady=(20, 0), sticky="w")
        self.name_label = ctk.CTkLabel(
            right, text=UNKNOWN_NAME,
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#4A9EFF", anchor="w", justify="left")
        self.name_label.grid(row=1, column=0, padx=20, pady=(0, 16), sticky="ew")

        ctk.CTkLabel(right, text="Эмоция:", font=ctk.CTkFont(size=14)).grid(
            row=2, column=0, padx=20, pady=(0, 0), sticky="w")
        self.emotion_label = ctk.CTkLabel(
            right, text="—",
            font=ctk.CTkFont(size=28, weight="bold"),
            anchor="w", justify="left")
        self.emotion_label.grid(row=3, column=0, padx=20, pady=(0, 16), sticky="ew")

        self.status_label = ctk.CTkLabel(
            right, text="Загрузка моделей...",
            font=ctk.CTkFont(size=12),
            text_color="#AAAAAA", anchor="w", justify="left",
            wraplength=300)
        self.status_label.grid(row=4, column=0, padx=20, pady=(0, 12), sticky="ew")

        sep = ctk.CTkFrame(right, height=1, fg_color="#444444")
        sep.grid(row=5, column=0, padx=20, pady=8, sticky="ew")

        ctk.CTkLabel(right, text="Добавить нового человека:",
                     font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=6, column=0, padx=20, pady=(8, 4), sticky="w")

        self.name_entry = ctk.CTkEntry(
            right, placeholder_text="Введите имя...")
        self.name_entry.grid(row=7, column=0, padx=20, pady=4, sticky="ew")

        self.add_btn = ctk.CTkButton(
            right, text="📸 Сохранить лицо",
            command=self._add_face, state="disabled")
        self.add_btn.grid(row=8, column=0, padx=20, pady=6, sticky="ew")

        self.rebuild_btn = ctk.CTkButton(
            right, text="🔄 Пересобрать базу из фото",
            fg_color="#555555", hover_color="#444444",
            command=self._rebuild_db, state="disabled")
        self.rebuild_btn.grid(row=9, column=0, padx=20, pady=(0, 20), sticky="ew")

        self._init_recognizer_async()
        self._update_loop()

    # ------------------------------------------------------------------
    def _init_recognizer_async(self):
        def worker():
            try:
                self.recognizer = FaceRecognizer()
                self._recognizer_ready = True
                self.after(0, lambda: self.status_label.configure(
                    text="Готово. Покажите лицо в камеру."))
                self.after(0, lambda: self.add_btn.configure(state="normal"))
                self.after(0, lambda: self.rebuild_btn.configure(state="normal"))
            except Exception as e:
                msg = f"Ошибка загрузки: {e}"
                self.after(0, lambda m=msg: self.status_label.configure(text=m))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    def _add_face(self):
        name = self.name_entry.get().strip()
        if not name:
            self.status_label.configure(text="Введите имя!")
            return
        if not self._recognizer_ready:
            self.status_label.configure(text="Модели ещё грузятся...")
            return

        frame, _ = self.camera.get_frame()
        if frame is None:
            self.status_label.configure(text="Кадр не получен")
            return

        # Сохраняем в фоне — UI не фризится
        self.status_label.configure(text=f"Сохранение '{name}'...")
        self.add_btn.configure(state="disabled")

        def worker():
            try:
                ok = self.recognizer.add_face(name, frame.copy())
                if ok:
                    msg = f"Лицо '{name}' сохранено"
                    self.after(0, lambda: self.name_entry.delete(0, "end"))
                else:
                    msg = "Лицо не найдено в кадре"
                self.after(0, lambda m=msg: self.status_label.configure(text=m))
            except Exception as e:
                msg = f"Ошибка: {e}"
                self.after(0, lambda m=msg: self.status_label.configure(text=m))
            finally:
                self.after(0, lambda: self.add_btn.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _rebuild_db(self):
        if not self._recognizer_ready:
            return
        self.status_label.configure(text="Пересборка базы...")
        self.rebuild_btn.configure(state="disabled")

        def worker():
            try:
                self.recognizer.rebuild_db()
                msg = "База пересобрана"
            except Exception as e:
                msg = f"Ошибка: {e}"
            self.after(0, lambda m=msg: self.status_label.configure(text=m))
            self.after(0, lambda: self.rebuild_btn.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    def _update_loop(self):
        if not self.winfo_exists():
            return

        frame, _ = self.camera.get_frame()

        if frame is not None and self._recognizer_ready and not self._analyzing:
            self._analyzing = True
            threading.Thread(
                target=self._analyze_async,
                args=(frame.copy(),),
                daemon=True,
            ).start()

        if frame is not None:
            # Рисуем рамку
            if self._last_box is not None:
                x, y, w, h = self._last_box
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (74, 158, 255), 2)

                # Русская подпись над рамкой через Pillow
                label = f"{self._last_name}"
                if self._last_emoji:
                    label = f"{self._last_name}  {self._last_emoji}"
                ty = max(10, y - 30)
                # Ограничим, чтобы не уходило за верх кадра
                if ty < 10:
                    ty = y + 5
                frame = draw_label(frame, label, x, ty)

            rgb = frame[:, :, ::-1]
            img = Image.fromarray(rgb)
            lw = max(self.video_label.winfo_width(), 320)
            lh = max(self.video_label.winfo_height(), 240)
            img = img.resize((lw, lh), Image.BILINEAR)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(lw, lh))
            self.video_label.configure(image=ctk_img, text="")
            self.video_label._image = ctk_img

        self.after(30, self._update_loop)

    def _analyze_async(self, frame):
        try:
            name, emo_en, emo_ru, emoji, box = self.recognizer.analyze(frame)
            self._last_name = name
            if emoji and emo_ru:
                self._last_emotion_ru = f"{emoji} {emo_ru}"
            elif emo_ru:
                self._last_emotion_ru = emo_ru
            else:
                self._last_emotion_ru = "—"
            self._last_emoji = emoji or ""
            self._last_box = box

            self.after(0, lambda n=name: self.name_label.configure(text=n))
            self.after(0, lambda t=self._last_emotion_ru:
                       self.emotion_label.configure(text=t))
        except Exception as e:
            print(f"[face_view] Ошибка анализа: {e}")
        finally:
            self._analyzing = False

    # ------------------------------------------------------------------
    def apply_settings(self, settings: dict):
        pass

    def on_hide(self):
        pass