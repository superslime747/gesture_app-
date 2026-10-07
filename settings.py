"""
Модуль сохранения и загрузки настроек приложения.
Настройки хранятся в JSON-файле settings.json рядом с приложением.
"""
import json
import os
from pathlib import Path

# Путь к файлу настроек
SETTINGS_FILE = Path(__file__).parent / "settings.json"

# Значения по умолчанию
DEFAULT_SETTINGS = {
    "camera_index": 0,       # индекс камеры
    "theme": "dark",         # "dark" | "light"
    "cursor_sensitivity": 5, # 1..10
    "smoothing": 5,          # 0..10
}


def load_settings() -> dict:
    """Загружает настройки из файла. Если файла нет — возвращает значения по умолчанию."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Дополняем недостающие ключи значениями по умолчанию
            for key, value in DEFAULT_SETTINGS.items():
                data.setdefault(key, value)
            return data
        except Exception as e:
            print(f"[settings] Ошибка чтения настроек: {e}")
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict) -> None:
    """Сохраняет настройки в файл."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[settings] Ошибка сохранения настроек: {e}")