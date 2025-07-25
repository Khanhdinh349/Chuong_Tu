# ui.py
import tkinter as tk
from tkinter import messagebox, font
import subprocess

window = tk.Tk()
window.title("SMART LOCKER SYSTEM")
window.geometry("600x800")
window.configure(bg="#E6F0FA")

title_font = font.Font(family="Tahoma", size=36, weight="bold")
button_font = font.Font(family="Tahoma", size=18)

tk.Label(window, text="SMART LOCKER", font=title_font, fg="blue", bg="#E6F0FA").pack(pady=20)

def run_deposit():
    messagebox.showinfo("Deposit", "Start face capture and training.")
    subprocess.run(["python", "train.py"])
    subprocess.run(["python", "realtime.py"])

def run_add():
    messagebox.showinfo("Add More", "Start face recognition to add more items.")
    subprocess.run(["python", "realtime.py"])

def run_retrieve():
    messagebox.showinfo("Retrieve", "Start face recognition to retrieve your item.")
    subprocess.run(["python", "realtime.py"])

tk.Button(window, text="📥 DEPOSIT", command=run_deposit, font=button_font, width=20, bg="#00BFFF", fg="white").pack(pady=30)
tk.Button(window, text="➕ ADD MORE", command=run_add, font=button_font, width=20, bg="#32CD32", fg="white").pack(pady=30)
tk.Button(window, text="📦 RETRIEVE", command=run_retrieve, font=button_font, width=20, bg="#FFA500", fg="white").pack(pady=30)

window.mainloop()
