import cv2
import time
import threading
import tkinter as tk
import customtkinter as cs
import pose_detector as pd
import semaphore_decoder as sd
from PIL import ImageTk, Image
from queue import Queue


class VideoThread(threading.Thread):
    def __init__(self, video_player, video_button, buffer_size=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cap = None
        self.aspect_ratio = 1
        self.video_player = video_player
        self.video_button = video_button
        self.buffer_size = buffer_size
        self.angle_buffer = Queue()
        self.not_paused_event = threading.Event()
        self.play_event = threading.Event()

    def run(self):
        detector = pd.PoseDetector()
        left_elbow = 13
        right_elbow = 14
        left_wrist = 15
        right_wrist = 16

        while True:
            if not self.play_event.is_set():
                default_img = Image.new("RGB", (1, 1), "black")
                self.update_video_player(default_img)
                self.play_event.wait()

            if not self.not_paused_event.is_set():
                self.not_paused_event.wait()

            success, img_bgr = self.cap.read()
            if success:
                img_bgr, landmarks = detector.find_pose(img_bgr)
                if len(detector.landmarks) != 0:
                    img_bgr, right_angle = detector.find_angle(img_bgr, right_elbow, right_wrist)
                    img_bgr, left_angle = detector.find_angle(img_bgr, left_elbow, left_wrist)
                    self.update_angle_buffer(right_angle, left_angle)
                else:
                    self.update_angle_buffer(None, None)

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_rgb = Image.fromarray(img_rgb)

                new_width = int(img_rgb.width * self.aspect_ratio)
                new_height = int(img_rgb.height * self.aspect_ratio)
                if new_width > 0 and new_height > 0:
                    img_rgb = img_rgb.resize((new_width, new_height))

                self.update_video_player(img_rgb)

            else:
                self.video_button.configure(state=tk.DISABLED)

    def set_cap(self, cap):
        self.cap = cap
        self.play_event.set()
        self.not_paused_event.set()

    def set_aspect_ratio(self, player_width, player_height):
        if self.cap is not None:
            image_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            image_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.aspect_ratio = min(player_width / image_width, player_height / image_height)

    def update_video_player(self, img):
        img = ImageTk.PhotoImage(image=img)
        self.video_player.config(image=img)
        self.video_player.image = img
        self.video_player.update_idletasks()

    def update_angle_buffer(self, right_angle, left_angle):
        self.angle_buffer.put((right_angle, left_angle))
        if self.angle_buffer.qsize() > self.buffer_size:
            self.angle_buffer.get()

    def toggle_pause(self):
        if self.not_paused_event.is_set():
            self.not_paused_event.clear()
            self.video_button.configure(text="Resume Video")
        else:
            self.not_paused_event.set()
            self.video_button.configure(text="Pause Video")

    def restart(self):
        if self.play_event.is_set():
            self.play_event.clear()
        if not self.not_paused_event.is_set():
            self.not_paused_event.set()

    def stop(self):
        if self.cap:
            self.cap.release()


class TextThread(threading.Thread):
    def __init__(self, detector_output, text_output, angle_buffer, buffer_size=10,
                 output_threshold=5, language="en", stable_duration=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector_output = detector_output
        self.text_output = text_output
        self.text = ""
        self.language = language
        self.buffer_size = buffer_size
        self.angle_buffer = angle_buffer
        self.letter_buffer = []
        self.output_threshold = output_threshold
        self.output_letter = None
        self.stable_letter = None
        self.stable_start_time = 0.0
        self.stable_duration = stable_duration
        self.start_detection_event = threading.Event()
        self.stable_letter_event = threading.Event()

    def run(self):
        decoder = sd.SemaphoreDecoder()
        self.start_detection_event.set()

        while True:
            if not self.start_detection_event.is_set():
                self.start_detection_event.wait()

            if self.angle_buffer:
                right_angle, left_angle = self.angle_buffer.get()
                current_letter = decoder.find_letter(right_angle, left_angle, self.language)
                self.update_letter_buffer(current_letter)

                if self.is_stable(current_letter):
                    if self.output_letter != current_letter:
                        self.stable_letter = None
                        self.output_letter = current_letter
                        self.stable_start_time = time.time()

                        if self.stable_letter_event.is_set():
                            self.stable_letter_event.clear()

                    elif time.time() - self.stable_start_time >= self.stable_duration:
                        self.stable_letter = current_letter

                        if not self.stable_letter_event.is_set():
                            self.stable_letter_event.set()
                            self.update_text()

                    self.output_letter = current_letter
                    self.update_output()

    def update_letter_buffer(self, letter):
        self.letter_buffer.append(letter)
        if len(self.letter_buffer) > self.buffer_size:
            self.letter_buffer.pop(0)

    def is_stable(self, letter):
        return self.letter_buffer.count(letter) >= self.output_threshold

    def update_output(self):
        default_color = self.detector_output.master.cget("fg_color")

        if self.output_letter is None:
            self.detector_output.configure(text=" ", fg_color=default_color)
        else:
            if self.stable_letter is None:
                self.detector_output.configure(fg_color=default_color)
            else:
                self.detector_output.configure(fg_color="orange")
            self.detector_output.configure(text=self.output_letter)

    def update_text(self):
        if self.stable_letter in ("STOP", "КІНЕЦЬ"):
            self.start_detection_event.clear()

        elif self.stable_letter in ("SPACE", "ПРОБІЛ"):
            self.text += " "
            self.text_output.configure(text=self.text)

        elif self.stable_letter is not None:
            self.text += self.stable_letter
            self.text_output.configure(text=self.text)

    def update_settings(self, language, stable_duration):
        self.language = language
        self.stable_duration = stable_duration

    def restart(self):
        self.text = ""
        self.letter_buffer = []
        self.output_letter = None
        self.stable_letter = None
        self.stable_start_time = 0.0

        if not self.start_detection_event.is_set():
            self.start_detection_event.set()
        if self.stable_letter_event.is_set():
            self.stable_letter_event.clear()

        self.text_output.configure(text=" ")
        self.detector_output.configure(text=" ")
        self.detector_output.configure(fg_color=self.detector_output.master.cget("fg_color"))


class SemaphoreApp(cs.CTk):
    def __init__(self, master=None):
        super().__init__(master)
        self.camera = None
        self.video_file = ''

        self.title("Semaphore Decoder")
        self.geometry(f"{1000}x{750}")

        self.grid_columnconfigure((2, 3), weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.settings_frame = cs.CTkFrame(self, corner_radius=15)
        self.settings_frame.grid(row=0, column=0, rowspan=4, columnspan=2, padx=(20, 10), pady=(20, 20), sticky="nsew")
        self.settings_frame.columnconfigure(0, weight=0)
        self.settings_frame.rowconfigure(3, weight=1)

        self.settings_label = cs.CTkLabel(self.settings_frame, text="Settings", font=cs.CTkFont(size=16, weight="bold"))
        self.settings_label.grid(row=0, column=0, columnspan=2, padx=(20, 10), pady=(20, 20), sticky="nsew")

        self.language_label = cs.CTkLabel(self.settings_frame, text="Language:")
        self.language_label.grid(row=1, column=0, padx=(20, 10), pady=(20, 20), sticky="nsew")
        self.language = cs.CTkOptionMenu(self.settings_frame, values=["English", "Ukrainian"])
        self.language.set("English")
        self.language.grid(row=1, column=1, padx=(10, 20), pady=(20, 20), sticky="nsew")

        self.detection_speed_label = cs.CTkLabel(self.settings_frame, text="Detection Speed:")
        self.detection_speed_label.grid(row=2, column=0, padx=(20, 10), pady=(20, 20), sticky="nsew")
        self.detection_speed = cs.CTkOptionMenu(self.settings_frame, values=["0.5 sec", "1 sec", "1.5 sec", "2 sec"])
        self.detection_speed.set("2 sec")
        self.detection_speed.grid(row=2, column=1, padx=(10, 20), pady=(20, 20), sticky="nsew")

        self.settings_button = cs.CTkButton(self.settings_frame, text="Update", command=self.on_settings_update)
        self.settings_button.grid(row=3, column=0, columnspan=2, padx=(20, 20), pady=(20, 20), sticky="s")

        self.video_frame = cs.CTkFrame(self, corner_radius=15)
        self.video_frame.grid(row=0, column=2, rowspan=3, columnspan=2, padx=(10, 20), pady=(20, 20), sticky="nsew")
        self.video_frame.columnconfigure((2, 3), weight=1)
        self.video_frame.rowconfigure(1, weight=1)

        self.button1 = cs.CTkButton(self.video_frame, text="Start Camera", command=self.start_camera)
        self.button1.grid(row=0, column=2, padx=20, pady=10, sticky="e")
        self.button2 = cs.CTkButton(self.video_frame, text="Open Video", command=self.open_video)
        self.button2.grid(row=0, column=3, padx=20, pady=10, sticky="w")

        self.video_player = tk.Label(self.video_frame, bg="black")
        self.video_player.grid(row=1, column=2, columnspan=2, padx=60, pady=(0, 20), sticky="nsew")

        self.detector_output = cs.CTkLabel(self.video_frame, text=" ", width=72, height=72,
                                           font=cs.CTkFont(size=36, weight="bold"),
                                           corner_radius=15)
        self.detector_output.grid(row=2, column=2, columnspan=2, padx=20, pady=(0, 20))

        self.text_frame = cs.CTkFrame(self, corner_radius=15)
        self.text_frame.grid(row=3, column=2, columnspan=2, padx=(10, 20), pady=(0, 20), sticky="nsew")
        self.text_frame.columnconfigure(2, weight=1)
        self.text_frame.rowconfigure(0, weight=1)

        self.text_output = cs.CTkLabel(self.text_frame, text=" ", height=128,
                                       font=cs.CTkFont(size=36, weight="bold"),
                                       wraplength=800, corner_radius=15)
        self.text_output.grid(row=0, column=2, columnspan=2, padx=20, pady=(20, 20), sticky="nsew")

        self.video_player.bind("<Configure>", self.on_resize)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        buffer_size = 10
        self.video_thread = VideoThread(video_player=self.video_player, video_button=self.button2,
                                        buffer_size=buffer_size,
                                        daemon=True)
        self.video_thread.start()

        self.text_thread = TextThread(detector_output=self.detector_output, text_output=self.text_output,
                                      buffer_size=buffer_size, angle_buffer=self.video_thread.angle_buffer,
                                      daemon=True)
        self.text_thread.start()

    def start_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1600)
        self.button1.configure(text="Restart", command=self.restart)
        self.button2.configure(state=tk.DISABLED)

        self.video_thread.set_cap(self.camera)
        self.on_resize(None)

    def open_video(self):
        self.video_file = tk.filedialog.askopenfilename(
            filetypes=[('Video', ['*.mp4', '*.avi', '*.mov', '*.mkv', '*gif']), ('All Files', '*.*')])
        if self.video_file:
            self.button1.configure(text="Restart", command=self.restart)
            self.button2.configure(text="Pause Video", command=self.video_thread.toggle_pause)

            self.video_thread.set_cap(cv2.VideoCapture(self.video_file))
            self.on_resize(None)

    def restart(self):
        self.video_thread.restart()
        self.text_thread.restart()
        self.button1.configure(text="Start Camera", command=self.start_camera)
        self.button2.configure(text="Open Video", command=self.open_video)
        self.button2.configure(state=tk.NORMAL)

    def on_settings_update(self):
        language_dict = {
            "English": "en",
            "Ukrainian": "uk"
        }
        detection_speed_dict = {
            "0.5 sec": 0.5,
            "1 sec": 1.0,
            "1.5 sec": 1.5,
            "2 sec": 2.0
        }
        language = language_dict[self.language.get()]
        detection_speed = detection_speed_dict[self.detection_speed.get()]

        self.text_thread.update_settings(language, detection_speed)

    def on_close(self):
        if self.camera:
            self.camera.release()
        self.video_thread.stop()
        self.destroy()

    def on_resize(self, event):
        player_width = self.video_player.winfo_width()
        player_height = self.video_player.winfo_height()
        self.video_thread.set_aspect_ratio(player_width, player_height)
        self.text_output.configure(wraplength=self.text_frame.winfo_width() * 0.7)


if __name__ == "__main__":
    cs.set_appearance_mode("dark")
    cs.set_default_color_theme("dark-blue")
    app = SemaphoreApp()
    app.mainloop()
