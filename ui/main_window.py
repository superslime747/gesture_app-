"""
Главное окно приложения.
Содержит верхнюю панель, центральную область (динамическую) и нижнюю панель
с переключателем режимов.
"""
import customtkinter as ctk

from settings import load_settings, save_settings
from camera import CameraStream
from classifier import GestureClassifier
from ui.game_view import GameView
from ui.mouse_view import MouseView
from ui.face_view import FaceView
from ui.settings_view import SettingsView


class MainWindow(ctk.CTk):
    """Главное окно приложения."""

    def __init__(self):
        super().__init__()

        # Настройки
        self.settings = load_settings()
        self._apply_theme()

        self.title("Gesture App — распознавание жестов")
        self.geometry("1200x720")
        self.minsize(1000, 600)

        # Общие ресурсы
        self.camera = None
        self.classifier = GestureClassifier()

        # Текущий экран
        self.current_view = None

        # Сетка
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        # Верхняя панель
        self._build_topbar()

        # Центральная область
        self.center_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.center_frame.grid(row=1, column=0, sticky="nsew", padx=16, pady=8)
        self.center_frame.grid_rowconfigure(0, weight=1)
        self.center_frame.grid_columnconfigure(0, weight=1)

        # Нижняя панель
        self._build_bottombar()

        # Стартовый экран
        self.show_mode_selection()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Тема
    # ------------------------------------------------------------------
    def _apply_theme(self):
        theme = self.settings.get("theme", "dark")
        ctk.set_appearance_mode("dark" if theme == "dark" else "light")
        ctk.set_default_color_theme("blue")

    # ------------------------------------------------------------------
    # Верхняя панель
    # ------------------------------------------------------------------
    def _build_topbar(self):
        self.topbar = ctk.CTkFrame(self, height=50, corner_radius=0)
        self.topbar.grid(row=0, column=0, sticky="ew")
        self.topbar.grid_columnconfigure(1, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.topbar, text="✋ Gesture App",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=16, pady=8, sticky="w")

        self.settings_btn = ctk.CTkButton(
            self.topbar, text="⚙ Настройки", width=140,
            command=self.show_settings
        )
        self.settings_btn.grid(row=0, column=2, padx=16, pady=8, sticky="e")

    # ------------------------------------------------------------------
    # Нижняя панель
    # ------------------------------------------------------------------
    def _build_bottombar(self):
        self.bottombar = ctk.CTkFrame(self, height=60, corner_radius=0)
        self.bottombar.grid(row=2, column=0, sticky="ew")
        self.bottombar.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.mode_btn_game = ctk.CTkButton(
            self.bottombar, text="🎮 Игра КНБ", height=40,
            command=self.show_game
        )
        self.mode_btn_game.grid(row=0, column=0, padx=8, pady=10)

        self.mode_btn_mouse = ctk.CTkButton(
            self.bottombar, text="🖱 Виртуальная мышь", height=40,
            command=self.show_mouse
        )
        self.mode_btn_mouse.grid(row=0, column=1, padx=8, pady=10)

        self.mode_btn_face = ctk.CTkButton(
            self.bottombar, text="😀 Лицо и эмоции", height=40,
            command=self.show_face
        )
        self.mode_btn_face.grid(row=0, column=2, padx=8, pady=10)

        self.mode_btn_home = ctk.CTkButton(
            self.bottombar, text="🏠 Главное меню", height=40,
            command=self.show_mode_selection
        )
        self.mode_btn_home.grid(row=0, column=3, padx=8, pady=10)

    # ------------------------------------------------------------------
    # Управление видами
    # ------------------------------------------------------------------
    def _clear_center(self):
        if self.current_view is not None:
            try:
                self.current_view.on_hide()
            except Exception:
                pass
            self.current_view.destroy()
            self.current_view = None

    def _set_view(self, view):
        self._clear_center()
        self.current_view = view
        view.grid(row=0, column=0, sticky="nsew")

    def show_mode_selection(self):
        """Экран выбора режима — три большие кнопки."""
        self._clear_center()
        frame = ctk.CTkFrame(self.center_frame, fg_color="transparent")
        frame.grid_rowconfigure((0, 1, 2), weight=1)
        frame.grid_columnconfigure((0, 1, 2), weight=1)

        title = ctk.CTkLabel(
            frame, text="Выбери режим",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=3, pady=(20, 10))

        btn_game = ctk.CTkButton(
            frame, text="🎮\n\nИгра\nКамень-Ножницы-Бумага",
            font=ctk.CTkFont(size=18, weight="bold"),
            height=200, width=260, corner_radius=20,
            command=self.show_game
        )
        btn_game.grid(row=1, column=0, padx=15, pady=20)

        btn_mouse = ctk.CTkButton(
            frame, text="🖱\n\nВиртуальная\nмышь",
            font=ctk.CTkFont(size=18, weight="bold"),
            height=200, width=260, corner_radius=20,
            command=self.show_mouse
        )
        btn_mouse.grid(row=1, column=1, padx=15, pady=20)

        btn_face = ctk.CTkButton(
            frame, text="😀\n\nЛицо\nи эмоции",
            font=ctk.CTkFont(size=18, weight="bold"),
            height=200, width=260, corner_radius=20,
            command=self.show_face
        )
        btn_face.grid(row=1, column=2, padx=15, pady=20)

        self.current_view = frame

    def show_game(self):
        if not self._ensure_camera():
            return
        view = GameView(self.center_frame, self.camera, self.classifier, self)
        self._set_view(view)

    def show_mouse(self):
        if not self._ensure_camera():
            return
        view = MouseView(self.center_frame, self.camera, self.classifier, self)
        self._set_view(view)

    def show_face(self):
        if not self._ensure_camera():
            return
        view = FaceView(self.center_frame, self.camera, self.classifier, self)
        self._set_view(view)

    def show_settings(self):
        view = SettingsView(self.center_frame, self.settings, self)
        self._set_view(view)

    # ------------------------------------------------------------------
    # Камера
    # ------------------------------------------------------------------
    def _ensure_camera(self) -> bool:
        if self.camera is not None and self.camera.running:
            return True
        try:
            self.camera = CameraStream(
                camera_index=self.settings.get("camera_index", 0)
            )
            self.camera.start()
            return True
        except Exception as e:
            self._show_error(f"Не удалось открыть камеру: {e}")
            self.camera = None
            return False

    def restart_camera(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
        self._ensure_camera()

    def _show_error(self, message: str):
        top = ctk.CTkToplevel(self)
        top.title("Ошибка")
        top.geometry("400x150")
        top.transient(self)
        top.grab_set()

        lbl = ctk.CTkLabel(top, text=message, wraplength=360,
                           font=ctk.CTkFont(size=14))
        lbl.pack(padx=20, pady=20, expand=True)

        btn = ctk.CTkButton(top, text="OK", command=top.destroy)
        btn.pack(pady=10)

    # ------------------------------------------------------------------
    # Применение настроек
    # ------------------------------------------------------------------
    def apply_settings(self, new_settings: dict):
        old_camera = self.settings.get("camera_index")
        self.settings.update(new_settings)
        save_settings(self.settings)

        self._apply_theme()

        if old_camera != self.settings.get("camera_index"):
            self.restart_camera()

        if self.current_view is not None and hasattr(self.current_view, "apply_settings"):
            try:
                self.current_view.apply_settings(self.settings)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _on_close(self):
        if self.camera is not None:
            self.camera.stop()
        self.destroy()