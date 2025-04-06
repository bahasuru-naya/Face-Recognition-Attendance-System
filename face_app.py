import cv2
import mediapipe as mp
import os
import shutil
import pandas as pd
import numpy as np
import json
import logging
import datetime
import random
import threading
import queue
import matplotlib.pyplot as plt
import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox, ttk
from tkcalendar import DateEntry
from PIL import Image, ImageTk
from settings.settings import PATHS, FACE_DETECTION, CAMERA, TRAINING, MEDIAPIPE_CONFIG, RECOGNITION_CONFIG

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split


# Directory to save registered faces
if not os.path.exists("registered_faces"):
    os.makedirs("registered_faces")

# CSV file to store attendance
csv_file = "attendance.csv"
if not os.path.exists(csv_file):
    pd.DataFrame(columns=["Username", "Date", "Time"]).to_csv(csv_file, index=False)


def load_settings():
    try:
        with open("settings.json", "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Settings file not found.")
        return {
            "attendance_time": {"start_time": "12:00", "end_time": "12:00"},
            "attendance_days": {day: False for day in
                                ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]},
            "automatic_training": True
        }


def save_settings(hour_var1, minute_var1, hour_var2, minute_var2, day_vars, t_bool):
    settings_data = {
        "attendance_time": {
            "start_time": f"{hour_var1.get()}:{minute_var1.get()}",
            "end_time": f"{hour_var2.get()}:{minute_var2.get()}"
        },
        "attendance_days": {day: var.get() for day, var in day_vars.items()},
        "automatic_training": t_bool.get()
    }

    with open("settings.json", "w") as file:
        json.dump(settings_data, file, indent=4)

    messagebox.showinfo("Success", f"Settings saved successfully.")


def settings():
    settings_data = load_settings()

    S_window = tk.Toplevel(root)
    S_window.title("Settings")
    center_window(S_window, 600, 650)
    S_window.resizable(False, False)
    S_window.configure(bg="#ffffff")
    settings_section = tk.LabelFrame(S_window, text="Attendance taking time range", font=("Segoe UI", 14, "bold"),
                                     bg="#ffffff", padx=10, pady=10, relief=tk.RIDGE, borderwidth=2)
    settings_section.pack(pady=10, padx=20, fill=tk.X)

    time1_label = tk.Label(settings_section, text="Start Time:", bg="#ffffff", font=("Segoe UI", 10))
    time1_label.pack()

    start_hour, start_minute = map(int, settings_data["attendance_time"]["start_time"].split(":"))
    end_hour, end_minute = map(int, settings_data["attendance_time"]["end_time"].split(":"))

    hour_var1 = tk.StringVar(value=str(start_hour))
    minute_var1 = tk.StringVar(value=str(start_minute))
    hour_var2 = tk.StringVar(value=str(end_hour))
    minute_var2 = tk.StringVar(value=str(end_minute))

    time_frame1 = tk.Frame(settings_section)
    time_frame1.pack()
    hour_spinbox1 = tk.Spinbox(time_frame1, from_=0, to=23, textvariable=hour_var1, wrap=True, width=5)
    hour_spinbox1.pack(side=tk.LEFT)
    tk.Label(time_frame1, text=":", bg="#ffffff").pack(side=tk.LEFT)
    minute_spinbox1 = tk.Spinbox(time_frame1, from_=0, to=59, textvariable=minute_var1, wrap=True, width=5)
    minute_spinbox1.pack(side=tk.LEFT)
    tk.Label(time_frame1, text="(24 hour format)", bg="#ffffff").pack(side=tk.LEFT)

    time2_label = tk.Label(settings_section, text="End Time:", bg="#ffffff", font=("Segoe UI", 10))
    time2_label.pack()
    time_frame2 = tk.Frame(settings_section)
    time_frame2.pack()
    hour_spinbox2 = tk.Spinbox(time_frame2, from_=00, to=23, textvariable=hour_var2, wrap=True, width=5)
    hour_spinbox2.pack(side=tk.LEFT)
    tk.Label(time_frame2, text=":", bg="#ffffff").pack(side=tk.LEFT)
    minute_spinbox2 = tk.Spinbox(time_frame2, from_=00, to=59, textvariable=minute_var2, wrap=True, width=5)
    minute_spinbox2.pack(side=tk.LEFT)
    tk.Label(time_frame2, text="(24 hour format)", bg="#ffffff").pack(side=tk.LEFT)

    settings_section1 = tk.LabelFrame(S_window, text="Attendance taking dates", font=("Segoe UI", 14, "bold"),
                                      bg="#ffffff", padx=10, pady=10, relief=tk.RIDGE, borderwidth=2)
    settings_section1.pack(pady=10, padx=20, fill=tk.X)

    days_of_week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    day_vars = {day: tk.BooleanVar(value=settings_data["attendance_days"].get(day, False)) for day in days_of_week}

    for day, var in day_vars.items():
        checkbox = ctk.CTkCheckBox(settings_section1, text=day,
                                   variable=var, font=("Segoe UI", 15),
                                   text_color="#2d3436", border_color="#dfe6e9")
        checkbox.pack(anchor="w", padx=5, pady=5)

    settings_train_section = tk.LabelFrame(S_window, text="Automatic Training", font=("Segoe UI", 14, "bold"),
                                           bg="#ffffff", padx=10, pady=10, relief=tk.RIDGE, borderwidth=2)
    settings_train_section.pack(pady=10, padx=20, fill=tk.X)

    t_bool = tk.BooleanVar(value=settings_data["automatic_training"])

    t_checkbox = ctk.CTkCheckBox(settings_train_section, text="Automatic Training ",
                                 variable=t_bool, font=("Segoe UI", 15),
                                 text_color="#2d3436", border_color="#dfe6e9")
    t_checkbox.pack(anchor="w", padx=5, pady=5)
    t_label = tk.Label(settings_train_section, text="This take more time when "
                                                    "registering or deleting a face", bg="#ffffff",
                       font=("Segoe UI", 10))
    t_label.pack(anchor="w")

    submit_button = ctk.CTkButton(S_window, text="Save", font=("Segoe UI", 20), height=35,
                                  command=lambda: save_settings(hour_var1, minute_var1, hour_var2, minute_var2,
                                                                day_vars, t_bool))
    submit_button.pack(pady=5)


def augment_face_images(user_folder):
    images = [f for f in os.listdir(user_folder) if f.endswith('.jpg')]
    for image_name in images:
        image_path = os.path.join(user_folder, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Augmentation 1: Horizontal flip
        flipped = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(user_folder, f"{image_name[:-4]}_flip.jpg"), flipped)

        # Augmentation 2: Brightness adjustment
        brightness_factor = random.uniform(0.5, 1.5)
        bright_img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
        cv2.imwrite(os.path.join(user_folder, f"{image_name[:-4]}_bright.jpg"), bright_img)

        # Augmentation 3: Zoom effect
        zoom_factor = random.uniform(1.1, 1.3)
        zoomed_img = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

        zh, zw, _ = zoomed_img.shape
        startx = (zh - img.shape[0]) // 2
        starty = (zw - img.shape[1]) // 2
        zoom_cropped = zoomed_img[startx:startx + img.shape[0], starty:starty + img.shape[1]]
        cv2.imwrite(os.path.join(user_folder, f"{image_name[:-4]}_zoom.jpg"), zoom_cropped)

        # Augmentation 4: Rotate
        angle = random.randint(-10, 10)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, rot_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        cv2.imwrite(os.path.join(user_folder, f"{image_name[:-4]}_rotate.jpg"), rotated)


def register_face(username):

    if not username:
        messagebox.showerror("Error", "Please enter a username to register.")
        return

    user_folder = os.path.join("registered_faces", username)
    if os.path.exists(user_folder):
        messagebox.showerror("Error", f"Username '{username}' already registered. Please use a different username.")
        return

    try:
        os.makedirs(user_folder)
        print(f"Created directory: {user_folder}")
    except OSError as e:
        messagebox.showerror("Error", f"Could not create directory for registration:\n{e}")
        return

    mp_face_detection = mp.solutions.face_detection
    try:
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=MEDIAPIPE_CONFIG['min_detection_confidence']
        )
    except Exception as e:
        messagebox.showerror("Error", f"Failed to initialize MediaPipe Face Detection:\n{e}")
        if os.path.exists(user_folder):
            try: shutil.rmtree(user_folder)
            except OSError as clean_e: print(f"Warning: Failed to clean up directory {user_folder}: {clean_e}")
        return

    cam = initialize_camera(CAMERA['index'])
    if cam is None:
        messagebox.showerror("Error", f"Could not open webcam index {CAMERA['index']}")
        face_detection.close()
        if os.path.exists(user_folder):
            try: shutil.rmtree(user_folder)
            except OSError as clean_e: print(f"Warning: Failed to clean up directory {user_folder}: {clean_e}")
        return

    register_window = tk.Toplevel(root)
    register_window.title(f"Registering Face for {username}")
    center_window(register_window, 700, 600)
    register_window.resizable(False, False)
    register_window.configure(bg="#ffffff")
    register_window.grab_set()
    progress_label = tk.Label(register_window, text="Initializing Camera...", font=("Segoe UI", 14, "bold"),
                              fg="#333", bg="#ffffff")
    progress_label.pack(pady=(10, 2))
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(register_window, variable=progress_var, maximum=TRAINING['samples_needed'], mode='determinate')
    progress_bar.pack(fill="x", padx=20, pady=5)
    video_label = tk.Label(register_window, bg="#ffffff")
    video_label.pack(pady=10)

    count = 0
    pause_capture = False
    capture_active = True
    registration_started = False

    def update_frame():
        nonlocal count, pause_capture, capture_active, registration_started

        if not capture_active:
            return

        if pause_capture:
            register_window.after(100, update_frame)
            return

        ret, frame = cam.read()
        if not ret:
            print("Warning: Failed to grab frame from camera.")
            if capture_active:
                register_window.after(50, update_frame)
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = face_detection.process(rgb_frame)
        rgb_frame.flags.writeable = True
        frame_display = frame.copy()

        if not registration_started:
             progress_label.config(text="Waiting for user confirmation...")
        else:
             progress_label.config(text=f"Capturing Samples for {username}")

        face_detected_in_frame = False

        if registration_started and results.detections:
            detection = results.detections[0]
            face_detected_in_frame = True

            bboxC = detection.location_data.relative_bounding_box
            xmin = int(bboxC.xmin * w)
            ymin = int(bboxC.ymin * h)
            width_face = int(bboxC.width * w)
            height_face = int(bboxC.height * h)
            pad = MEDIAPIPE_CONFIG['padding']
            x1 = max(0, xmin - pad)
            y1 = max(0, ymin - pad)
            x2 = min(w, xmin + width_face + pad)
            y2 = min(h, ymin + height_face + pad)

            cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if count < TRAINING['samples_needed']:
                if x2 > x1 and y2 > y1:
                    face_roi = frame[y1:y2, x1:x2]
                    if face_roi.size != 0:
                        face_img_resized = cv2.resize(face_roi, FACE_DETECTION['face_img_size'], interpolation=cv2.INTER_LINEAR)
                        img_path = os.path.join(user_folder, f"{count + 1}.jpg")
                        try:
                            cv2.imwrite(img_path, face_img_resized)
                            count += 1
                            progress_var.set(count)
                            print(f"Saved: {img_path} ({count}/{TRAINING['samples_needed']})")
                        except Exception as e:
                            print(f"Error saving image {img_path}: {e}")
                    else: print("Warning: Face ROI calculation resulted in empty image.")
                else: print("Warning: Invalid crop dimensions calculated.")

        if registration_started:
            progress_text = f"Captured: {count}/{TRAINING['samples_needed']}"
            text_color = (0, 255, 0) if face_detected_in_frame else (0, 0, 255)
        else:
            progress_text = "Waiting to start..."
            text_color = (255, 150, 0) # Orange for waiting state

        cv2.putText(frame_display, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)

        if registration_started:
            if not face_detected_in_frame and count < TRAINING['samples_needed']:
                cv2.putText(frame_display, "No face detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
             cv2.putText(frame_display, "Click OK on message box to start...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2, cv2.LINE_AA)


        img_display_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_display_rgb)
        imgtk = ImageTk.PhotoImage(image=image)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk, bg="#ffffff")


        if registration_started and count >= TRAINING['samples_needed']:
            finish_registration()
            return

        if capture_active:
            register_window.after(20, update_frame)

    def finish_registration():
        nonlocal capture_active
        if not capture_active: return
        capture_active = False
        print("Finishing registration...")
        progress_label.config(text="Registration Complete!")
        if cam and cam.isOpened(): cam.release(); print("Camera released.")
        if 'face_detection' in locals(): face_detection.close(); print("MediaPipe resources released.")
        try:
            #augment_face_images(user_folder)
            settings_data = load_settings()
            if settings_data.get("automatic_training", False): train_face_model()
        except Exception as e:
             print(f"Error during post-processing: {e}")
             messagebox.showwarning("Warning", f"Registration complete, error during post-processing:\n{e}", parent=register_window)
        register_window.after(1000, lambda: [
            register_window.destroy(),
            messagebox.showinfo("Success", f"Face registered successfully for {username} with {count} images."),
            refresh_combobox()
        ])

    def on_close():
        nonlocal pause_capture, capture_active, registration_started # Include flag
        pause_capture = True
        response = messagebox.askyesno("Stop Registration",
                                       "Are you sure you want to stop registering?\n"
                                       "All captured images for this user will be deleted.",
                                       parent=register_window)
        if response:
            capture_active = False
            registration_started = False
            print("Registration cancelled by user.")
            if cam and cam.isOpened(): cam.release(); print("Camera released.")
            if 'face_detection' in locals(): face_detection.close(); print("MediaPipe resources released.")
            if os.path.exists(user_folder):
                try:
                    shutil.rmtree(user_folder)
                    print(f"Removed incomplete registration folder: {user_folder}")
                except OSError as e:
                    print(f"Error removing folder {user_folder}: {e}")
                    messagebox.showerror("Error", f"Could not remove directory:\n{e}\nPlease remove it manually.", parent=register_window)
            register_window.destroy()
        else:
            pause_capture = False
            print("Resuming registration view.")
            # update_frame()

    def show_info_and_start():
        nonlocal registration_started
        messagebox.showinfo("Info",
                            "Look at the camera to register your face.\n"
                            "Move your head slightly to capture different angles.\n"
                            "Make sure only the person being registered is visible in the video feed.\n"
                            "Click OK to start capturing images.",
                            parent=register_window)

        if capture_active:
            registration_started = True
            print("User clicked OK. Starting capture...")
            progress_label.config(text=f"Capturing Samples for {username}")


    register_window.protocol("WM_DELETE_WINDOW", on_close)
    update_frame()
    register_window.after(100, show_info_and_start)


def delete_face(username, del_record):
    if username:
        user_folder = os.path.join("registered_faces", username)

        # Delete user folder if it exists
        if os.path.exists(user_folder):
            try:
                shutil.rmtree(user_folder)
                messagebox.showinfo("Success", f"Deleted folder for user '{username}'")
                settings_data = load_settings()
                t_bool = settings_data["automatic_training"]
                if t_bool is True:
                    train_face_model()

            except Exception as e:
                messagebox.showerror("Error", f"Error deleting folder: {e}")
        else:
            messagebox.showinfo("Error", "Username directory does not exist")

        # Delete record in attendance.csv if requested
        if del_record is True:
            attendance_file = "attendance.csv"
            if os.path.exists(attendance_file):
                try:
                    df = pd.read_csv(attendance_file)
                    if username in df['Username'].values:
                        df = df[df['Username'] != username]
                        df.to_csv(attendance_file, index=False)
                        messagebox.showinfo("Success", f"Deleted attendance record for '{username}'")
                    else:
                        messagebox.showinfo("Info", f"No attendance record found for '{username}'")
                except Exception as e:
                    messagebox.showerror("Error", f"Error updating attendance.csv: {e}")
            else:
                messagebox.showinfo("Error", "attendance.csv does not exist")


def refresh_combobox():
    folders = [folder for folder in os.listdir("registered_faces") if
               os.path.isdir(os.path.join("registered_faces", folder))]
    user_combobox['values'] = folders


def delete_user():
    selected_user = user_combobox.get()
    if not selected_user:
        messagebox.showwarning("Warning", "Please select a username.")
        return
    confirm = messagebox.askyesno("Confirm", f"Are you sure you want to delete '{selected_user}'?")
    if confirm:
        delete_face(selected_user, delete_checkbox_var.get())
        user_combobox.set("")
        refresh_combobox()


# Function to recognize faces and log attendance
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def initialize_camera(camera_index: int = 0) -> cv2.VideoCapture:
    try:
        cam = cv2.VideoCapture(camera_index)
        if not cam.isOpened():
            logger.error("Could not open webcam")
            return None

        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA['width'])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA['height'])
        return cam
    except Exception as e:
        logger.error(f"Error initializing camera: {e}")
        return None


def load_names(filename: str) -> dict:
    try:
        names_json = {}
        if os.path.exists(filename):
            with open(filename, 'r') as fs:
                content = fs.read().strip()
                if content:
                    names_json = json.loads(content)
        return names_json
    except Exception as e:
        logger.error(f"Error loading names: {e}")
        return {}


def recognize_faces():
    recognition_active = True
    cam = None
    face_detection = None
    worker_thread = None
    frame_queue = queue.Queue(maxsize=2)

    try:
        logger.info("Initializing face recognition system with MediaPipe...")
        mp_face_detection = mp.solutions.face_detection
        try:
            face_detection = mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=MEDIAPIPE_CONFIG['min_detection_confidence']
            )
            logger.info("MediaPipe Face Detection initialized.")
        except Exception as e:
            raise ValueError(f"Failed to initialize MediaPipe Face Detection: {e}")

        # Initialize camera
        cam = initialize_camera(CAMERA['index'])
        if cam is None:
            raise ValueError(f"Failed to initialize camera index {CAMERA['index']}")
        logger.info("Camera initialized.")

        # Load names and model
        names = load_names(PATHS['names_file'])
        if not names:
            logger.warning("Names file not loaded or empty. Recognition will only show indices or 'Unknown'.")

        try:
            model = load_model(PATHS['model_file'])
            logger.info(f"CNN model loaded from {PATHS['model_file']}")
        except Exception as e:
            raise ValueError(f"Failed to load model '{PATHS['model_file']}': {e}")


        def recognition_worker():
            nonlocal recognition_active # Allow modification of the flag
            logger.info("Recognition worker thread started.")
            while recognition_active:
                if not cam or not cam.isOpened():
                    logger.error("Camera is not opened in worker thread.")
                    recognition_active = False
                    break

                ret, frame = cam.read()
                if not ret:
                    logger.warning("Failed to grab frame from camera.")
                    time.sleep(0.1)
                    continue

                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = face_detection.process(rgb_frame)
                rgb_frame.flags.writeable = True
                display_frame = frame.copy()

                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        xmin = int(bboxC.xmin * w)
                        ymin = int(bboxC.ymin * h)
                        width_face = int(bboxC.width * w)
                        height_face = int(bboxC.height * h)
                        pad = RECOGNITION_CONFIG['padding']

                        crop_x1 = max(0, xmin - pad)
                        crop_y1 = max(0, ymin - pad)
                        crop_x2 = min(w, xmin + width_face + pad)
                        crop_y2 = min(h, ymin + height_face + pad)

                        rect_x1, rect_y1, rect_x2, rect_y2 = crop_x1, crop_y1, crop_x2, crop_y2
                        cv2.rectangle(display_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), 2)

                        # --- Prepare face ROI for model prediction ---
                        if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                            face_roi_rgb = rgb_frame[crop_y1:crop_y2, crop_x1:crop_x2]

                            if face_roi_rgb.size != 0:

                                face_img_resized = cv2.resize(face_roi_rgb, RECOGNITION_CONFIG['model_input_size'], interpolation=cv2.INTER_LINEAR)
                                # Expand dimensions for batch (1, height, width, channels)
                                face_img_expanded = np.expand_dims(face_img_resized, axis=0)

                                # --- CNN Prediction ---
                                try:
                                    predictions = model.predict(face_img_expanded, verbose=0) # verbose=0 prevents console spam
                                    class_index = np.argmax(predictions[0])
                                    confidence = predictions[0][class_index]

                                    # --- Display Results ---
                                    name = "Unknown"
                                    attendance_mark_label = ""
                                    if confidence >= RECOGNITION_CONFIG['confidence_threshold']:
                                        name = names.get(str(class_index), f"Index {class_index}")
                                        attendance_mark_label = log_attendance(name)
                                        name_color = (255, 255, 255)

                                    else:
                                        name = "Unknown"
                                        name_color = (200, 200, 200)
                                        attendance_mark_label = ""

                                    # Put text on display frame
                                    if name:
                                        cv2.putText(display_frame, name, (rect_x1 + 5, rect_y1 - 7),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, name_color, 2)
                                    confidence_text = f"{confidence:.4f}"
                                    cv2.putText(display_frame, confidence_text, (rect_x1 + 5, rect_y2 + 20),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                                    if attendance_mark_label:
                                        cv2.putText(display_frame, attendance_mark_label, (rect_x1 + 5, rect_y2 + 45),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                                except Exception as pred_e:
                                    logger.error(f"Error during model prediction: {pred_e}")
                                    cv2.putText(display_frame, "Pred Error", (rect_x1 + 5, rect_y1 - 7),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                            else:
                                logger.warning("Face ROI calculation resulted in empty image during crop.")
                        else:
                            logger.warning("Invalid crop dimensions calculated.")


                # --- Send frame to UI thread ---
                try:
                    rgb_display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    if frame_queue.full():
                        try: frame_queue.get_nowait()
                        except queue.Empty: pass
                    frame_queue.put(rgb_display_frame)
                except Exception as q_e:
                    logger.error(f"Error putting frame into queue: {q_e}")

            logger.info("Recognition worker thread finished.")
            if cam and cam.isOpened():
                 cam.release()
                 logger.info("Camera released by worker thread.")

        # --- Tkinter Window Setup ---
        video_window = tk.Toplevel(root)
        video_window.title("Face Recognition Feed")
        center_window(video_window, 700, 550)
        video_window.resizable(False, False)
        video_window.configure(bg="#ffffff")
        video_label = tk.Label(video_window, bg="#ffffff")
        video_label.pack(pady=10, padx=10)


        def update_video_label():
            if not recognition_active:
                return
            try:
                frame_to_display = frame_queue.get_nowait()
                image = Image.fromarray(frame_to_display)
                imgtk = ImageTk.PhotoImage(image=image)
                video_label.imgtk = imgtk
                video_label.configure(image=imgtk, bg="#ffffff")

            except queue.Empty:
                pass
            except Exception as ui_e:
                 logger.error(f"Error updating Tkinter video label: {ui_e}")

            if recognition_active:
                video_label.after(1, update_video_label)

        # --- Window Close Handler ---
        def on_close():
            nonlocal recognition_active, worker_thread
            logger.info("Close button clicked. Stopping recognition...")
            recognition_active = False

            if worker_thread is not None and worker_thread.is_alive():
                logger.info("Waiting for worker thread to join...")
                worker_thread.join(timeout=1.0)
                if worker_thread.is_alive():
                     logger.warning("Worker thread did not join within timeout.")

            if cam is not None and cam.isOpened():
                cam.release()
                logger.info("Camera released by on_close handler.")
            if face_detection is not None:
                face_detection.close()
                logger.info("MediaPipe resources released by on_close handler.")

            video_window.destroy()
            logger.info("Recognition window closed.")

        video_window.protocol("WM_DELETE_WINDOW", on_close)

        # --- Start Threads ---
        worker_thread = threading.Thread(target=recognition_worker, daemon=True)
        worker_thread.start()
        update_video_label()
        logger.info("Face recognition system running...")


    except Exception as e:
        logger.error(f"An error occurred during recognition setup: {e}", exc_info=True)
        messagebox.showerror("Recognition Error", f"Failed to start recognition:\n{e}", parent=root)
        recognition_active = False
        if worker_thread is not None and worker_thread.is_alive(): worker_thread.join(timeout=0.5)
        if cam is not None and cam.isOpened(): cam.release()
        if face_detection is not None: face_detection.close()
        if 'video_window' in locals() and video_window.winfo_exists():
             video_window.destroy()

# Function to log attendance
def log_attendance(username):
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    current_day = now.strftime("%A")
    df = pd.read_csv(csv_file)

    settings_data = load_settings()
    start_time = settings_data["attendance_time"]["start_time"]
    end_time = settings_data["attendance_time"]["end_time"]

    start_time_f = f"{int(start_time.split(':')[0]):02}:{int(start_time.split(':')[1]):02}:00"
    end_time_f = f"{int(end_time.split(':')[0]):02}:{int(end_time.split(':')[1]):02}:00"

    attendance_days = settings_data["attendance_days"]

    # Check if attendance allowed for current day
    if not attendance_days.get(current_day, False):
        return "not allowed on today"

    # Check if current time is in attendance time range
    elif not (start_time_f <= time <= end_time_f):
        return "time out of range"

    elif ((df['Username'] == username) & (df['Date'] == date)).any():
        return "attendence - marked"

    elif not ((df['Username'] == username) & (df['Date'] == date)).any():
        new_entry = pd.DataFrame({"Username": [username], "Date": [date], "Time": [time]})
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(csv_file, index=False)
        return "attendence - marked"

    else:
        return "attendence - not marked"


# train the model
def update_names_json(path: str):
    # Get list of all user folders
    user_folders = [os.path.join(path, folder) for folder in os.listdir(path) if
                    os.path.isdir(os.path.join(path, folder))]

    if not user_folders:
        messagebox.showerror("Error", "No user folders found for training.")
        return 0  # Return 0 if no folders found

    # Assign ID based on folder index
    id_username_map = {str(idx): os.path.basename(folder) for idx, folder in enumerate(user_folders, start=0)}

    # Write to JSON file
    with open(PATHS['names_file'], 'w') as f:
        json.dump(id_username_map, f, indent=4)

    return len(user_folders)


class FaceDataset:
    def __init__(self, dataDir):
        self.data_dir = dataDir

    def dataPaths(self):
        filepaths = []
        labels = []
        folds = os.listdir(self.data_dir)
        for fold in folds:
            foldPath = os.path.join(self.data_dir, fold)
            filelist = os.listdir(foldPath)
            for file in filelist:
                fpath = os.path.join(foldPath, file)
                filepaths.append(fpath)
                labels.append(fold)
        return filepaths, labels  # independent and dependent variable

    def dataFrame(self, files, labels):
        Fseries = pd.Series(files, name='filepaths')
        Lseries = pd.Series(labels, name='labels')
        return pd.concat([Fseries, Lseries], axis=1)

    def split_(self, train_size=0.8):
        files, labels = self.dataPaths()
        df = self.dataFrame(files, labels)
        strat = df['labels']
        trainData, validData = train_test_split(df, train_size=train_size, shuffle=True, random_state=42, stratify=strat)
        return trainData, validData


def get_image_generators(path: str, img_size=(224, 224), batch_size=32):

    dataSplit = FaceDataset(path)
    train_data, valid_data = dataSplit.split_(train_size=0.8)

    color = 'rgb'

    # Define ImageDataGenerators with data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.5, 1.5],
        zoom_range = 0.3)

    valid_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_dataframe(  # withou increse number of images
        train_data,
        x_col='filepaths',
        y_col='labels',
        target_size=img_size,
        color_mode=color,
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    print("Shape of augmented training images:", train_generator.image_shape)

    valid_generator = valid_datagen.flow_from_dataframe(
        valid_data,
        x_col='filepaths',
        y_col='labels',
        target_size=img_size,
        color_mode=color,
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    print("Shape of validation images:", valid_generator.image_shape)


    return train_generator, valid_generator


def create_model(input_shape, num_classes):

    classifier_vgg16 = tf.keras.applications.VGG16(input_shape=input_shape,include_top=False,weights='imagenet')

    # not train top layers
    for layer in classifier_vgg16.layers:
        layer.trainable = False

    # adding extra layers for our class/images
    main_model = classifier_vgg16.output

    main_model = Conv2D(512, (3, 3), activation='relu', padding='same')(main_model)
    main_model = MaxPooling2D(pool_size=(2, 2))(main_model)

    main_model = Conv2D(512, (3, 3), activation='relu', padding='same')(main_model)
    main_model = MaxPooling2D(pool_size=(2, 2))(main_model)

    main_model = GlobalAveragePooling2D()(main_model)
    main_model = Dense(1024, activation='relu')(main_model)
    main_model = Dense(1024, activation='relu')(main_model)
    main_model = Dense(512, activation='relu')(main_model)
    main_model = Dense(num_classes, activation='softmax')(main_model)

    model = Model(inputs=classifier_vgg16.input, outputs=main_model)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model


class TrainingProgressCallback(Callback):
    def __init__(self, progress_bar, status_label, window):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_label = status_label
        self.window = window
        self.total_epochs = 0
        self.batches_per_epoch = 0
        self.total_batches = 0
        self.current_epoch = 0

    def on_train_begin(self, logs=None):
        # Called once at the beginning of training
        try:
            self.total_epochs = self.params['epochs']
            self.batches_per_epoch = self.params.get('steps')

            if self.batches_per_epoch is None:
                 print("Warning: 'steps' (batches per epoch) not found in callback params. Progress calculation might be inaccurate.")
                 self.batches_per_epoch = 1 # Avoid division by zero, but progress will be wrong

            self.total_batches = self.total_epochs * self.batches_per_epoch
            self.progress_bar['maximum'] = 100 # Keep maximum as 100 for percentage
            print(f"Training started: {self.total_epochs} epochs, {self.batches_per_epoch} batches/epoch, {self.total_batches} total batches.")
        except KeyError as e:
            print(f"Error accessing training parameters in callback: {e}. Progress bar may not function correctly.")
            self.total_epochs = 1
            self.batches_per_epoch = 1
            self.total_batches = 1
        except Exception as e:
            print(f"Unexpected error in on_train_begin: {e}")
            self.total_epochs = 1
            self.batches_per_epoch = 1
            self.total_batches = 1

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        if self.current_epoch == 0:
            self.status_label.config(text=f"Epoch {epoch + 1}/{self.total_epochs} starting...")
        else:
            self.status_label.config(text=f"Epoch {epoch}/{self.total_epochs} completed. \n "
                                          f"Epoch {epoch + 1}/{self.total_epochs} starting...")
        self.window.update_idletasks()


    def on_train_batch_end(self, batch, logs=None):
        if self.total_batches > 0: # Avoid division by zero if params weren't read

            # Calculate overall progress
            batches_completed = self.current_epoch * self.batches_per_epoch + (batch + 1)
            progress_percent = int((batches_completed / self.total_batches) * 100)
            self.progress_bar['value'] = progress_percent

            # Update status label with batch info (optional: add batch loss/acc from logs)
            batch_loss = logs.get('loss', 'N/A')
            batch_acc = logs.get('accuracy', 'N/A')
            loss_str = f"{batch_loss:.4f}" if isinstance(batch_loss, float) else batch_loss
            acc_str = f"{batch_acc:.4f}" if isinstance(batch_acc, float) else batch_acc

            status_text = (f"Epoch {self.current_epoch + 1}/{self.total_epochs} | "
                           f"Batch {batch + 1}/{self.batches_per_epoch}\n"
                           f"Overall Progress: {progress_percent}% | Batch Loss: {loss_str}, Batch Acc: {acc_str}")
            self.status_label.config(text=status_text)

            # Updating the UI every batch can slow down training significantly.
            # Consider updating less frequently (e.g., every 10 batches) if needed:
            # if (batch + 1) % 10 == 0:
            #     self.window.update_idletasks()
            self.window.update_idletasks() # Force UI update every batch (as requested)
        else:
            # Fallback status if total_batches wasn't calculated
            self.status_label.config(text=f"Epoch {self.current_epoch + 1}, Batch {batch + 1} processed...")
            self.window.update_idletasks()


def train_face_model():
    # --- Setup Training Window ---
    training_window = tk.Toplevel(root)
    training_window.title("Training Model")
    # Adjusted size slightly for potentially longer status text
    center_window(training_window, 450, 160)
    training_window.resizable(False, False)
    training_label = tk.Label(training_window, text="Training model, please wait...", font=("Segoe UI", 12))
    training_label.pack(pady=10)

    # Progress Bar
    progress_bar = ttk.Progressbar(training_window, orient='horizontal', length=400, mode='determinate', maximum=100)
    progress_bar.pack(pady=10, padx=20)

    # Status Label
    # Allow label to wrap if text gets long
    status_label = tk.Label(training_window, text="Initializing training...", justify=tk.LEFT, font=("Segoe UI", 9), wraplength=400)
    status_label.pack(pady=5, padx=20, fill=tk.X)

    # Make the window appear immediately
    training_window.update_idletasks()
    training_window.update()

    try:
        # --- Data Preparation ---
        status_label.config(text="Updating names and counting faces...")
        training_window.update_idletasks()
        ids = update_names_json(PATHS['image_dir'])

        n_faces = ids
        if n_faces is None or n_faces == 0:
             raise ValueError("Could not determine the number of faces. Check image directory and names.json.")

        print(f"Number of faces (classes): {n_faces}")
        status_label.config(text=f"Found {n_faces} user(s). Preparing data generators...")
        training_window.update_idletasks()

        train, test = get_image_generators(PATHS['image_dir'])

        if not train or not test:
            training_window.destroy()
            messagebox.showwarning("No Data", "No valid face data found to train the model. Ensure images exist and are structured correctly.")
            return

        print("Input image shape:", train.image_shape)
        status_label.config(text="Data generators ready. Creating model...")
        training_window.update_idletasks()

        # --- Model Creation ---
        model = create_model((224, 224, 3), n_faces)

        # --- Training ---
        epochs = 5
        batch_size = 32 # Ensure this matches how the generator is configured if steps not auto-detected

        # Create the callback instance (no need to pass epochs explicitly anymore)
        progress_callback = TrainingProgressCallback(progress_bar, status_label, training_window)

        status_label.config(text="Starting model training...")
        training_window.update_idletasks()

        history = model.fit(train,
                            # batch_size=batch_size, # Often batch_size is inferred from generator
                            epochs=epochs,
                            validation_data=test,
                            shuffle=True,
                            callbacks=[progress_callback]) # Pass the callback here

        # --- Post-Training ---
        status_label.config(text=f"Training complete. \n "
                                 f"Saving model and generating training graphs...")
        training_window.update_idletasks()
        model.save('models/trained_model.h5')

        save_dir = "training_graphs"
        os.makedirs(save_dir, exist_ok=True)
        train_accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure()
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training vs Validation Loss')
        plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
        plt.close()

        plt.figure()
        plt.plot(train_accuracy, label='Training Accuracy')
        plt.plot(val_accuracy, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training vs Validation Accuracy')
        plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'))
        plt.close()

        print("Graphs saved successfully in", save_dir)

        # --- Finalize ---
        training_window.destroy()
        messagebox.showinfo("Training Complete", f"Model trained successfully for {n_faces} face(s).\n"
                                                 f"Training graphs are saved in '{save_dir}' folder.")

    except Exception as e:
        if 'training_window' in locals() and training_window.winfo_exists():
            training_window.destroy()
        messagebox.showerror("Error", f"An error occurred during training: {e}")
        print(f"Training Error: {e}") # Also print to console for debugging
        import traceback
        traceback.print_exc() # Print full traceback for debugging complex errors


# Function to display attendance in UI
def display_attendance(s_date):
    # Get the selected date from the DateEntry widget
    selected_date = s_date.get_date().strftime("%Y-%m-%d")

    settings_data = load_settings()
    start_time = settings_data["attendance_time"]["start_time"]
    end_time = settings_data["attendance_time"]["end_time"]

    start_time_f = f"{int(start_time.split(':')[0]):02}:{int(start_time.split(':')[1]):02}:00"
    end_time_f = f"{int(end_time.split(':')[0]):02}:{int(end_time.split(':')[1]):02}:00"

    # Create a new window for video feed
    attendance_window1 = tk.Toplevel(root)
    attendance_window1.title(f"Attendance on {selected_date}")
    center_window(attendance_window1, 600, 460)
    attendance_window1.resizable(False, False)
    attendance_window1.configure(bg="#ffffff")

    header_label_attendance = tk.Label(attendance_window1, text=f"Attendance on {selected_date}",
                                       font=("Segoe UI", 16, "bold"),
                                       fg="#333", bg="#ffffff")
    header_label_attendance.pack(pady=10, padx=10)

    # Attendance Labels
    attendance_frame1 = tk.Frame(attendance_window1, bg="#ffffff", padx=10, pady=10, relief=tk.RIDGE, borderwidth=2)
    attendance_frame1.pack(pady=10, padx=20, fill=tk.X)

    attendees_label1 = tk.Label(attendance_frame1, text="Present: ", font=("Segoe UI", 12, "bold"), bg="#ffffff",
                                fg="#2fa341")
    attendees_label1.pack(pady=2, anchor="w")

    absentees_label1 = tk.Label(attendance_frame1, text="Absent: ", font=("Segoe UI", 12, "bold"), bg="#ffffff",
                                fg="#c20202")
    absentees_label1.pack(pady=2, anchor="w")

    # Attendance Table
    tree_frame1 = tk.Frame(attendance_window1)
    tree_frame1.pack(pady=30, padx=20, fill=tk.BOTH, expand=True)

    tree1 = ttk.Treeview(tree_frame1, columns=("Username", "Date", "Time"), show='headings', height=10)
    tree1.heading("Username", text="Username")
    tree1.heading("Date", text="Date")
    tree1.heading("Time", text="Time")

    tree1.column("Username", width=140, anchor="center")
    tree1.column("Date", width=140, anchor="center")
    tree1.column("Time", width=140, anchor="center")

    tree1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Add scrollbar
    scrollbar = ttk.Scrollbar(tree_frame1, orient="vertical", command=tree1.yview)
    tree1.configure(yscroll=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill="y")

    df = pd.read_csv(csv_file)
    df = df[df['Date'] == selected_date]
    df = df[(df['Time'] >= start_time_f) & (df['Time'] <= end_time_f)]

    attendees1 = df[(df['Time'] >= start_time_f) & (df['Time'] <= end_time_f)]['Username'].unique()
    total_users1 = os.listdir("registered_faces")
    absentees1 = set(total_users1) - set(attendees1)
    attendees_label1.config(text=f"Present: {', '.join(attendees1)}")
    absentees_label1.config(text=f"Absent: {', '.join(absentees1)}")

    for i in tree1.get_children():
        tree1.delete(i)
    for _, row in df.iterrows():
        tree1.insert("", "end", values=(row["Username"], row["Date"], row["Time"]))


def all_display_attendance():
    # Create a new window for video feed
    attendance_window = tk.Toplevel(root)
    attendance_window.title("All Attendance Data")
    center_window(attendance_window, 600, 480)
    attendance_window.resizable(False, False)
    attendance_window.configure(bg="#ffffff")

    header_label_attendance_all = tk.Label(attendance_window, text=f"All Attendance Data",
                                           font=("Segoe UI", 16, "bold"),
                                           fg="#333", bg="#ffffff")
    header_label_attendance_all.pack(pady=10, padx=10)

    # Attendance Table
    tree_frame = tk.Frame(attendance_window)
    tree_frame.pack(pady=30, padx=20, fill=tk.BOTH, expand=True)

    tree = ttk.Treeview(tree_frame, columns=("Username", "Date", "Time"), show='headings', height=10)
    tree.heading("Username", text="Username")
    tree.heading("Date", text="Date")
    tree.heading("Time", text="Time")

    tree.column("Username", width=140, anchor="center")
    tree.column("Date", width=140, anchor="center")
    tree.column("Time", width=140, anchor="center")

    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Add scrollbar
    scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill="y")

    df = pd.read_csv(csv_file)
    for i in tree.get_children():
        tree.delete(i)
    for _, row in df.iterrows():
        tree.insert("", "end", values=(row["Username"], row["Date"], row["Time"]))


def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y - 50}")


# UI Setup
root = tk.Tk()
root.title("Face Recognition Attendance System")
center_window(root, 750, 710)
root.resizable(False, False)
root.configure(bg="#f7f7f7")

# Header
head_frame = tk.Frame(root, bg="#4a90e2")
head_frame.pack(fill=tk.X)

header_label = tk.Label(head_frame, text="Face Recognition Attendance System", font=("Segoe UI", 18, "bold"),
                        bg="#4a90e2", fg="white")
header_label.pack(side=tk.LEFT, pady=15, padx=15)

settings_btn = ctk.CTkButton(head_frame, text="⚙️ Settings", font=("Segoe UI", 16), height=30, command=settings)
settings_btn.pack(side=tk.RIGHT, padx=20, pady=10)

# Registration & Deletion Section
section1 = tk.Frame(root, bg="#f7f7f7")
section1.pack(pady=5, fill=tk.X, padx=20)

# Register Face Frame
register_frame = tk.LabelFrame(section1, text="Register Face", font=("Segoe UI", 14, "bold"), bg="#ffffff", fg="#333",
                               padx=15, pady=15)
register_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

username_label = tk.Label(register_frame, text="Username:", font=("Segoe UI", 12), bg="#ffffff")
username_label.pack(anchor="w")

username_entry = ctk.CTkEntry(register_frame, font=("Segoe UI", 16), width=235, height=40, fg_color="#ffffff",
                              corner_radius=10, border_color="#333", text_color="#333")
username_entry.pack(pady=15, fill=tk.X)

register_btn = ctk.CTkButton(register_frame, text="✅ Register", font=("Segoe UI", 18), height=40, fg_color="#2fa341",
                             hover_color="#237a31", command=lambda: register_face(username_entry.get()))
register_btn.pack(pady=20, fill=tk.X)

# Delete Face Frame
delete_frame = tk.LabelFrame(section1, text="Delete Face", font=("Segoe UI", 14, "bold"), bg="#ffffff", fg="#333",
                             padx=15, pady=15)
delete_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)

delete_username_label = tk.Label(delete_frame, text="Username:", font=("Segoe UI", 12), bg="#ffffff")
delete_username_label.pack(anchor="w")

user_combobox = ttk.Combobox(delete_frame, font=("Segoe UI", 12), state="readonly", width=23)
user_combobox.pack(pady=5, fill=tk.X)
user_combobox['values'] = []
refresh_combobox()

delete_checkbox_var = tk.BooleanVar()
delete_checkbox = ctk.CTkCheckBox(delete_frame, text="Delete with Records",
                                  variable=delete_checkbox_var, font=("Segoe UI", 15),
                                  text_color="#2d3436", border_color="#dfe6e9")
delete_checkbox.pack(anchor="w", pady=10)

delete_btn = ctk.CTkButton(delete_frame, text="❎ Delete", font=("Segoe UI", 18), height=40, fg_color="#c20202",
                           hover_color="#960000", command=delete_user)
delete_btn.pack(pady=10, fill=tk.X)

# Recognition & Attendance Section
section2 = tk.Frame(root, bg="#f7f7f7")
section2.pack(pady=5, fill=tk.X, padx=20)

# Mark Attendance Frame
mark_frame = tk.LabelFrame(section2, text="Mark Attendance", font=("Segoe UI", 14, "bold"), bg="#ffffff", fg="#333",
                           padx=15, pady=15)
mark_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

recognize_btn = ctk.CTkButton(mark_frame, text="🔎 Recognize Faces", font=("Segoe UI", 18), height=40, width=260,
                              command=recognize_faces)
recognize_btn.pack(pady=30, padx=20, fill=tk.X)

# Attendance Display Frame
attendance_frame = tk.LabelFrame(section2, text="Show Attendance", font=("Segoe UI", 14, "bold"), bg="#ffffff",
                                 fg="#333", padx=15, pady=15)
attendance_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

date_frame = tk.Frame(attendance_frame, bg="#ffffff")
date_frame.pack(pady=5, anchor="w")

start_date_label = tk.Label(date_frame, text="Date:", font=("Segoe UI", 12), bg="#ffffff")
start_date_label.pack(side=tk.LEFT)

start_date = DateEntry(date_frame, font=("Segoe UI", 12), width=8)
start_date.pack(side=tk.LEFT, padx=10)

attendance_btn = ctk.CTkButton(date_frame, text="📖 Show Attendance", font=("Segoe UI", 14), height=35,
                               command=lambda: display_attendance(start_date))
attendance_btn.pack(pady=10, fill=tk.X)

all_btn = ctk.CTkButton(attendance_frame, text="📚 Show All Records", font=("Segoe UI", 16), height=35,
                        command=all_display_attendance)
all_btn.pack(pady=5, fill=tk.X)

# Training Section
training_frame = tk.LabelFrame(root, text="Train the Model", font=("Segoe UI", 14, "bold"), bg="#ffffff", fg="#333",
                               padx=15, pady=15)
training_frame.pack(pady=10, padx=30, fill=tk.X)

training_ins_label = tk.Label(training_frame,
                              text="Click below to train the model manually.",
                              font=("Segoe UI", 11), bg="#ffffff")
training_ins_label.pack(pady=5, anchor="w")

train_btn = ctk.CTkButton(training_frame, text="🛠️ Train the Model", font=("Segoe UI", 18), height=40,
                          command=train_face_model)
train_btn.pack(pady=10, fill=tk.X)

root.mainloop()
