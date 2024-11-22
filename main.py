import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import math
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = None
push_up_goal = 10
push_up_type = "Standard"
count = 0
position = None
running = False
start_time = None
feedback_text = ""
push_ups_per_minute = 0


def calculate_angle(a, b, c):
    ab = [b[0] - a[0], b[1] - a[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    norm_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    norm_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    if norm_ab == 0 or norm_bc == 0:
        return 0
    angle = math.acos(dot_product / (norm_ab * norm_bc))
    return angle * (180.0 / math.pi)


class PushUpCheckerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Push-up Checker")
        self.root.geometry("1200x800")
        
        self.setup_ui()
        self.video_thread = None

    def setup_ui(self):
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()

        tk.Label(self.main_frame, text="Push-up Goal:", font=("Arial", 16)).pack(pady=10)
        self.goal_entry = tk.Entry(self.main_frame, font=("Arial", 16))
        self.goal_entry.pack()

        tk.Label(self.main_frame, text="Push-up Type:", font=("Arial", 16)).pack(pady=10)
        self.type_var = tk.StringVar(value="Standard")
        tk.OptionMenu(self.main_frame, self.type_var, "Standard", "Diamond", "Wide-arm").pack()

        self.start_button = tk.Button(self.main_frame, text="Start", font=("Arial", 16), command=self.start_pushup_checker)
        self.start_button.pack(pady=20)

        self.video_frame = tk.Frame(self.root)
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        self.feedback_label = tk.Label(self.video_frame, text="", font=("Arial", 20), fg="red")
        self.feedback_label.pack(pady=20)

        self.stats_label = tk.Label(self.video_frame, text="", font=("Arial", 16), fg="blue")
        self.stats_label.pack(pady=10)

        self.stop_button = tk.Button(self.video_frame, text="Stop", font=("Arial", 16), command=self.stop_pushup_checker)
        self.stop_button.pack(pady=10, side=tk.RIGHT)

    def start_pushup_checker(self):
        global running, push_up_goal, push_up_type, count, position, start_time

        try:
            push_up_goal = int(self.goal_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid push-up goal.")
            return

        push_up_type = self.type_var.get()
        count = 0
        position = None
        start_time = time.time()
        running = True

        self.main_frame.pack_forget()
        self.video_frame.pack()

        self.video_thread = threading.Thread(target=self.run_pushup_detection)
        self.video_thread.start()

    def stop_pushup_checker(self):
        global running
        running = False
        self.video_frame.pack_forget()
        self.main_frame.pack()

    def run_pushup_detection(self):
        global cap, count, position, feedback_text, push_ups_per_minute, running
        
        cap = cv2.VideoCapture(0)
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            while running:
                success, image = cap.read()
                if not success:
                    break

                image = cv2.resize(image, (640, 480))
                image = cv2.flip(image, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                lmlist = []
                if results.pose_landmarks:
                    h, w, _ = image.shape
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    for id, lm in enumerate(results.pose_landmarks.landmark):
                        x, y = int(lm.x * w), int(lm.y * h)
                        lmlist.append([id, x, y])

                feedback_text = "Perform a push-up!"
                if lmlist:
                    left_shoulder = lmlist[11][1:3]
                    left_elbow = lmlist[13][1:3]
                    left_wrist = lmlist[15][1:3]
                    right_shoulder = lmlist[12][1:3]
                    right_elbow = lmlist[14][1:3]
                    right_wrist = lmlist[16][1:3]

                    left_hip = lmlist[23][1:3]
                    right_hip = lmlist[24][1:3]
                    left_ankle = lmlist[27][1:3]
                    right_ankle = lmlist[28][1:3]

                    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    if left_arm_angle > 160 and right_arm_angle > 160:
                        position = "down"
                    elif left_arm_angle < 50 and right_arm_angle < 50 and position == "down":
                        if self.is_aligned(left_hip, right_hip, left_ankle, right_ankle):
                            position = "up"
                            count += 1
                            feedback_text = f"Push-ups: {count}"

                elapsed_time = time.time() - start_time
                push_ups_per_minute = count / (elapsed_time / 60) if elapsed_time > 0 else 0

                self.update_video(image)
                self.feedback_label.config(text=feedback_text)
                self.stats_label.config(text=f"Push-ups: {count}/{push_up_goal} | Rate: {push_ups_per_minute:.1f} per min")

                if count >= push_up_goal:
                    self.feedback_label.config(text="Goal Reached! Well Done!")
                    break

        cap.release()

    def is_aligned(self, left_hip, right_hip, left_ankle, right_ankle):
        """Check alignment of the hips and feet for proper form."""
        hip_alignment = abs(left_hip[1] - right_hip[1]) < 50
        feet_alignment = abs(left_ankle[1] - right_ankle[1]) < 50
        return hip_alignment and feet_alignment

    def update_video(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.video_label.config(image=img)
        self.video_label.image = img


if __name__ == "__main__":
    root = tk.Tk()
    app = PushUpCheckerApp(root)
    root.mainloop()
