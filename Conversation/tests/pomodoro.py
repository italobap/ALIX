import time
import keyboard

class PomodoroTimer:
    def __init__(self):
        self.short_pomodoro_timer = None
        self.long_pomodoro_timer = None
        self.absence_timer = None
        self.total_timer = None
        self.start_time = None

    def start_short_pomodoro(self):
        print("Short Pomodoro started for 10 seconds.")
        time.sleep(10)
        self.start_long_pomodoro()

    def start_long_pomodoro(self):
        if self.short_pomodoro_timer:
            self.short_pomodoro_timer.cancel()
        print("Long Pomodoro started for 20 seconds.")
        time.sleep(20)
        self.start_user_absence()

    def start_user_absence(self):
        if self.long_pomodoro_timer:
            self.long_pomodoro_timer.cancel()
        print("You've been absent for too long. Please return.")
        self.absence_timer = time.time()

    def start_total_timer(self):
        if self.absence_timer:
            self.absence_timer = None
        if not self.start_time:
            self.start_time = time.time()
            print("Press 'r' again to stop and calculate the total time.")

    def stop_and_calculate_total_time(self):
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            print(f"Total time between 'r' presses: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes).")
            self.start_time = None

    def start_listening(self):
        keyboard.on_press_key('r', self.on_key_press)
        keyboard.wait('esc')

    def on_key_press(self, e):
        if not self.start_time:
            self.start_short_pomodoro()
        else:
            self.stop_and_calculate_total_time()

if __name__ == '__main__':
    timer = PomodoroTimer()
    timer.start_listening()
