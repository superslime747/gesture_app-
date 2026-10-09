"""
Точка входа приложения.
"""
# ---- Глушим логи TensorFlow ДО всех импортов ----
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# -------------------------------------------------

from ui.main_window import MainWindow


def main():
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()