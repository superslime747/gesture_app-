"""
Логика игры "Камень-Ножницы-Бумага".
Игрок показывает жест — приложение распознаёт его через классификатор,
ИИ выбирает случайный жест, определяется победитель.
"""
import random
import time


# Метки жестов КНБ (должны совпадать с классификатором)
ROCK = 0
SCISSORS = 1
PAPER = 2

RPS_LABELS = {
    ROCK: "✊ Камень",
    SCISSORS: "✌ Ножницы",
    PAPER: "✋ Бумага",
}

# Что побеждает что: ключ побеждает значение
BEATS = {
    ROCK: SCISSORS,
    SCISSORS: PAPER,
    PAPER: ROCK,
}


class RPSGame:
    """Состояние игры КНБ."""

    def __init__(self):
        self.player_score = 0
        self.ai_score = 0
        self.player_move = None
        self.ai_move = None
        self.result_text = ""
        self.countdown_text = ""
        self.phase = "idle"  # idle | countdown | waiting | result
        self._countdown_start = 0.0
        self._wait_start = 0.0

    # ------------------------------------------------------------------
    # Счёт
    # ------------------------------------------------------------------
    def reset(self):
        self.player_score = 0
        self.ai_score = 0
        self.player_move = None
        self.ai_move = None
        self.result_text = ""
        self.countdown_text = ""
        self.phase = "idle"

    # ------------------------------------------------------------------
    # Раунд
    # ------------------------------------------------------------------
    def start_round(self):
        """Запускает отсчёт 3-2-1."""
        self.player_move = None
        self.ai_move = None
        self.result_text = ""
        self.phase = "countdown"
        self._countdown_start = time.time()

    def update(self, player_move_label):
        """
        Обновляет состояние игры.
        player_move_label — метка жеста (int) или None.
        Возвращает True, если раунд завершён.
        """
        now = time.time()

        if self.phase == "countdown":
            elapsed = now - self._countdown_start
            if elapsed < 1.0:
                self.countdown_text = "3"
            elif elapsed < 2.0:
                self.countdown_text = "2"
            elif elapsed < 3.0:
                self.countdown_text = "1"
            else:
                self.countdown_text = "ПОКАЗЫВАЙ!"
                self.phase = "waiting"
                self._wait_start = now

        elif self.phase == "waiting":
            # Ждём 0.5 сек — за это время фиксируем жест игрока
            if now - self._wait_start >= 0.5:
                # Игрок должен показать один из жестов КНБ
                if player_move_label in (ROCK, SCISSORS, PAPER):
                    self.player_move = player_move_label
                    self.ai_move = random.choice([ROCK, SCISSORS, PAPER])
                    self._resolve()
                    self.phase = "result"
                    return True
                # Иначе — не распознали, отменяем раунд
                self.result_text = "Не распознано — попробуй снова"
                self.phase = "idle"
                return True

        return False

    def _resolve(self):
        """Определяет победителя."""
        p, a = self.player_move, self.ai_move
        if p == a:
            self.result_text = "Ничья"
        elif BEATS[p] == a:
            self.result_text = "Победа!"
            self.player_score += 1
        else:
            self.result_text = "Проигрыш"
            self.ai_score += 1