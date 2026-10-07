"""
Экран настроек.
Камера, тема, чувствительность, сглаживание.
"""
import customtkinter as ctk

from camera import list_available_cameras


class SettingsView(ctk.CTkFrame):
    """Вид настроек."""

    def __init__(self, master, settings: dict, app):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.settings = settings.copy()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        # Заголовок
        title = ctk.CTkLabel(self, text="Настройки",
                             font=ctk.CTkFont(size=24, weight="bold"))
        title.grid(row=0, column=0, pady=(10, 20), sticky="w")

        # Форма
        form = ctk.CTkFrame(self, corner_radius=16)
        form.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        form.grid_columnconfigure(1, weight=1)

        # --- Камера ---
        ctk.CTkLabel(form, text="Камера:", font=ctk.CTkFont(size=15)).grid(
            row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        cameras = list_available_cameras()
        if not cameras:
            cameras = [0]
        cam_values = [f"Камера {i}" for i in cameras]
        current_cam = self.settings.get("camera_index", 0)
        if current_cam not in cameras:
            cameras.append(current_cam)
            cam_values.append(f"Камера {current_cam}")

        self.camera_menu = ctk.CTkOptionMenu(
            form, values=cam_values, width=200
        )
        self.camera_menu.grid(row=0, column=1, padx=20, pady=(20, 10), sticky="w")
        try:
            idx = cameras.index(current_cam)
            self.camera_menu.set(cam_values[idx])
        except ValueError:
            self.camera_menu.set(cam_values[0])
        self._camera_indices = cameras

        # --- Тема ---
        ctk.CTkLabel(form, text="Тема оформления:", font=ctk.CTkFont(size=15)).grid(
            row=1, column=0, padx=20, pady=10, sticky="w")

        theme_frame = ctk.CTkFrame(form, fg_color="transparent")
        theme_frame.grid(row=1, column=1, padx=20, pady=10, sticky="w")

        self.theme_var = ctk.StringVar(value=self.settings.get("theme", "dark"))
        rb_dark = ctk.CTkRadioButton(theme_frame, text="Тёмная",
                                     variable=self.theme_var, value="dark")
        rb_dark.pack(side="left", padx=(0, 20))
        rb_light = ctk.CTkRadioButton(theme_frame, text="Светлая",
                                      variable=self.theme_var, value="light")
        rb_light.pack(side="left")

        # --- Чувствительность ---
        ctk.CTkLabel(form, text="Чувствительность курсора:", font=ctk.CTkFont(size=15)).grid(
            row=2, column=0, padx=20, pady=10, sticky="w")

        self.sens_slider = ctk.CTkSlider(form, from_=1, to=10, number_of_steps=9)
        self.sens_slider.grid(row=2, column=1, padx=20, pady=10, sticky="ew")
        self.sens_slider.set(self.settings.get("cursor_sensitivity", 5))
        self.sens_value_lbl = ctk.CTkLabel(form, text=str(int(self.sens_slider.get())))
        self.sens_value_lbl.grid(row=2, column=2, padx=(0, 20))
        self.sens_slider.configure(command=lambda v: self.sens_value_lbl.configure(text=str(int(v))))

        # --- Сглаживание ---
        ctk.CTkLabel(form, text="Сглаживание движения:", font=ctk.CTkFont(size=15)).grid(
            row=3, column=0, padx=20, pady=10, sticky="w")

        self.smooth_slider = ctk.CTkSlider(form, from_=0, to=10, number_of_steps=10)
        self.smooth_slider.grid(row=3, column=1, padx=20, pady=10, sticky="ew")
        self.smooth_slider.set(self.settings.get("smoothing", 5))
        self.smooth_value_lbl = ctk.CTkLabel(form, text=str(int(self.smooth_slider.get())))
        self.smooth_value_lbl.grid(row=3, column=2, padx=(0, 20))
        self.smooth_slider.configure(command=lambda v: self.smooth_value_lbl.configure(text=str(int(v))))

        # --- Кнопки ---
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=2, column=0, pady=20, sticky="e")

        back_btn = ctk.CTkButton(btn_frame, text="Назад", width=120,
                                 command=self.app.show_mode_selection)
        back_btn.pack(side="left", padx=8)

        save_btn = ctk.CTkButton(btn_frame, text="Сохранить", width=120,
                                 command=self._save)
        save_btn.pack(side="left", padx=8)

    def _save(self):
        cam_text = self.camera_menu.get()
        try:
            idx = int(cam_text.split()[-1])
        except (ValueError, IndexError):
            idx = 0

        new_settings = {
            "camera_index": idx,
            "theme": self.theme_var.get(),
            "cursor_sensitivity": int(self.sens_slider.get()),
            "smoothing": int(self.smooth_slider.get()),
        }
        self.app.apply_settings(new_settings)
        self.app.show_mode_selection()

    def on_hide(self):
        pass