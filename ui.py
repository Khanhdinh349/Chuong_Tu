import os
import cv2
import time
import tkinter as tk
from tkinter import font, Button, SUNKEN, messagebox
from PIL import Image, ImageTk
import subprocess

# === Constants ===
LOCKER_COUNT = 10
AVAILABLE_FILE = "available.txt"
DATASET_DIR = "dataset"
CAPTURED_DIR = "captured"

# === Helper: Available Lockers State ===
def read_available_lockers():
    if os.path.exists(AVAILABLE_FILE):
        with open(AVAILABLE_FILE, "r") as f:
            return int(f.read().strip())
    return LOCKER_COUNT

def write_available_lockers(count):
    with open(AVAILABLE_FILE, "w") as f:
        f.write(str(count))

# === GUI Setup ===
window = tk.Tk()
window.title("SMART LOCKERS")
window.geometry("600x1024")
window.configure(bg="#BCD2EE")

# === Fonts ===
big_font_bold = font.Font(family="Tahoma", size=40, weight="bold")
normal_font_bold = font.Font(family="Tahoma", size=26, weight="bold")
normal_font = font.Font(family="Tahoma", size=20)
small_font = font.Font(family="Tahoma", size=14)

# === Header ===
canvas = tk.Canvas(window, width=600, height=128, bg="#FFFFFF")
canvas.place(x=0, y=0)
logo = Image.open("picture/logo.jpg").resize((400, 128))
header = ImageTk.PhotoImage(logo)
canvas.create_image(100, 0, image=header, anchor=tk.NW)

# === Title ===
tk.Label(window, text="SMART LOCKER", font=big_font_bold, fg="#191970", bg="#BCD2EE").pack(pady=(135, 0))
tk.Label(window, text="APPLYING FACE DETECTION & IOT", font=normal_font, fg="#666666", bg="#BCD2EE").pack()

# === Available Lockers ===
available_lockers = read_available_lockers()
locker_var = tk.StringVar(value=str(available_lockers))

def update_available_display():
    locker_var.set(str(read_available_lockers()))

def change_locker_count(delta):
    current = read_available_lockers()
    new_val = max(0, min(LOCKER_COUNT, current + delta))
    write_available_lockers(new_val)
    update_available_display()

frame = tk.Frame(window, bg="#BCD2EE", pady=10)
frame.pack()
tk.Label(frame, text="Available lockers:", font=normal_font_bold, fg="#B22222", bg="#BCD2EE").pack(side="left")
tk.Entry(frame, textvariable=locker_var, font=normal_font_bold, width=3, justify="center", fg="#B22222", bg="#FFFFE0").pack(side="left", padx=5)

# === SEND Function ===
def send_button():
    count = read_available_lockers()
    if count <= 0:
        messagebox.showwarning("Full", "All lockers are currently used.")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)
    idx = 1
    while os.path.exists(os.path.join(DATASET_DIR, str(idx))):
        idx += 1
    user_folder = os.path.join(DATASET_DIR, str(idx))
    os.makedirs(user_folder)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot access camera")
        return

    captured = 0
    total = 5
    start = time.time()
    interval = 3 / total

    while captured < total:
        ret, frame = cap.read()
        if not ret:
            continue
        if time.time() - start >= interval * captured:
            path = os.path.join(user_folder, f"image_{captured + 1}.jpg")
            cv2.imwrite(path, frame)
            captured += 1
            print(f"Saved: {path}")

    cap.release()
    change_locker_count(-1)
    messagebox.showinfo("Success", f"Images saved to {user_folder}")

    try:
        subprocess.run(["python", "train.py"], check=True)
        messagebox.showinfo("Training", "Training complete.")
    except Exception as e:
        messagebox.showerror("Train Error", str(e))

# === GET Function ===
def get_button():
    try:
        result = subprocess.run(["python", "realtime.py"], check=True)
    except subprocess.CalledProcessError:
        messagebox.showerror("Recognition", "Failed to recognize or no match found.")
        return

    os.makedirs(CAPTURED_DIR, exist_ok=True)
    idx = 1
    while os.path.exists(os.path.join(CAPTURED_DIR, str(idx))):
        idx += 1
    folder = os.path.join(CAPTURED_DIR, str(idx))
    os.makedirs(folder)

    # Capture one frame after successful match
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(os.path.join(folder, "image.jpg"), frame)
        messagebox.showinfo("GET", f"Image saved to {folder}")
        change_locker_count(+1)
    else:
        messagebox.showerror("Camera", "Could not capture image.")
    cap.release()

# === ADD Function ===
def add_button():
    try:
        subprocess.run(["python", "realtime.py"], check=True)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run realtime.py:\n{e}")

# === Button Creator ===
def create_image_button(x, y, img1_path, img2_path, command):
    img1 = ImageTk.PhotoImage(Image.open(img1_path))
    img2 = ImageTk.PhotoImage(Image.open(img2_path))
    btn = Button(window, image=img1, bg="#BCD2EE", border=0, cursor="hand2", command=command, relief=SUNKEN)
    btn.image1, btn.image2 = img1, img2
    btn.config(image=img1)

    def on_enter(e): btn.config(image=img2)
    def on_leave(e): btn.config(image=img1)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    btn.place(x=x, y=y)

# === Load Buttons ===
create_image_button(100, 340, "picture/BUTTON/SEND_1.png", "picture/BUTTON/SEND_2.png", send_button)
create_image_button(100, 520, "picture/BUTTON/ADD_1.png", "picture/BUTTON/ADD_2.png", add_button)
create_image_button(100, 700, "picture/BUTTON/GET_1.png", "picture/BUTTON/GET_2.png", get_button)

# === Footer ===
footer = tk.Frame(window, bg="#FFFFFF", pady=8)
footer.pack(fill="x", side="bottom")
tk.Label(footer, text="Pham Lu Huy Chuong - 2188201100\nTran Minh Thien - 2188200439", font=small_font, fg="#666666", bg="#FFFFFF", justify="center").pack()

window.mainloop()
