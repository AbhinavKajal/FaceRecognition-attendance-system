import os
import cv2
import numpy as np
import pandas as pd
import datetime
from tkinter import *
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Initialize
os.makedirs('dataset', exist_ok=True)
os.makedirs('trainer', exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

theme_mode = "light"
THEMES = {
    "light": {"bg": "#e8f4f8", "fg": "#333333", "btn": "#1e90ff", "btnfg": "white"},
    "dark": {"bg": "#2c2f33", "fg": "white", "btn": "#7289da", "btnfg": "white"},
}

def toggle_theme(root):
    global theme_mode
    theme_mode = "dark" if theme_mode == "light" else "light"
    apply_theme(root)


def apply_theme(widget):
    colors = THEMES[theme_mode]
    widget.configure(bg=colors["bg"])
    for child in widget.winfo_children():
        if isinstance(child, Button):
            child.configure(bg=colors["btn"], fg=colors["btnfg"], activebackground=colors["btn"])
        elif isinstance(child, Label):
            child.configure(bg=colors["bg"], fg=colors["fg"])
        elif isinstance(child, Frame):
            apply_theme(child)
        elif isinstance(child, Entry):
            child.configure(bg="white")

def capture_faces(user_id, name):
    cam = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, img = cam.read()
        if not ret: break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))  # Standardize face size
            cv2.imwrite(f"dataset/User.{user_id}.{count}.jpg", face)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(img, f"Image {count}/30", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Capturing Faces", img)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 30:
            break
    cam.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Done", f"{name}'s face data captured.")

def train_model():
    paths = [os.path.join("dataset", f) for f in os.listdir("dataset")]
    faces, ids = [], []
    for path in paths:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        id = int(path.split('.')[1])
        faces.append(gray)
        ids.append(id)
    recognizer.train(faces, np.array(ids))
    recognizer.save("trainer/trainer.yml")
    messagebox.showinfo("Success", "Model trained successfully.")

def recognize_and_mark(attendance_type="Entry", user_type="student"):
    recognizer.read("trainer/trainer.yml")
    df = pd.read_csv("users.csv") if os.path.exists("users.csv") else pd.DataFrame(columns=['ID', 'Name', 'Type'])
    cam = cv2.VideoCapture(0)
    already_marked = set()
    today = datetime.date.today().strftime("%Y-%m-%d")
    while True:
        ret, img = cam.read()
        if not ret: break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 45 and id_ not in already_marked:
                row = df[df["ID"] == id_]
                if not row.empty and row.iloc[0]["Type"] == user_type:
                    name = row.iloc[0]["Name"]
                    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    pd.DataFrame([[id_, name, now, attendance_type]]).to_csv("attendance.csv", mode='a', header=not os.path.exists("attendance.csv"), index=False)
                    messagebox.showinfo("Marked", f"{user_type.capitalize()} {name} marked for {attendance_type}")
                    already_marked.add(id_)
                    cam.release()
                    cv2.destroyAllWindows()
                    return
        cv2.imshow("Recognizing", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cam.release()
    cv2.destroyAllWindows()

def mark_absentees():
    today = datetime.date.today().strftime('%Y-%m-%d')
    df = pd.read_csv("users.csv") if os.path.exists("users.csv") else pd.DataFrame(columns=['ID', 'Name', 'Type'])
    df_today = pd.read_csv("attendance.csv") if os.path.exists("attendance.csv") else pd.DataFrame(columns=['ID','Name','DateTime','Type'])
    present_ids = df_today[df_today['DateTime'].str.startswith(today) & (df_today["Type"] == "Entry")]["ID"].unique()
    absentees = df[~df["ID"].isin(present_ids) & (df["Type"] == "student")]
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for _, row in absentees.iterrows():
        pd.DataFrame([[row['ID'], row['Name'], now, "Absent"]]).to_csv("attendance.csv", mode='a', header=False, index=False)
    messagebox.showinfo("Absentees", f"{len(absentees)} students marked absent.")

def teacher_portal():
    login = Toplevel()
    login.title("Teacher Login")
    login.geometry("300x200")
    apply_theme(login)

    Label(login, text="Username:").pack(pady=5)
    user_entry = Entry(login)
    user_entry.pack()

    Label(login, text="Password:").pack(pady=5)
    pass_entry = Entry(login, show='*')
    pass_entry.pack()

    def verify():
        if user_entry.get() == "admin" and pass_entry.get() == "1234":
            login.destroy()
            open_teacher_dashboard()
        else:
            messagebox.showerror("Error", "Invalid credentials")

    Button(login, text="Login", command=verify).pack(pady=15)

def open_teacher_dashboard():
    dash = Toplevel()
    dash.title("Teacher Dashboard")
    dash.geometry("700x600")
    apply_theme(dash)

    Button(dash, text="View Attendance", command=show_attendance).pack(pady=10)
    Button(dash, text="Auto Mark Absentees", command=mark_absentees).pack(pady=10)

    Label(dash, text="Delete Student", font=("Arial", 12, "bold")).pack(pady=10)
    del_frame = Frame(dash)
    del_frame.pack()
    apply_theme(del_frame)

    Label(del_frame, text="ID to Delete:").grid(row=0, column=0, padx=5)
    del_id = Entry(del_frame)
    del_id.grid(row=0, column=1)

    def delete_student():
        uid = del_id.get()
        if not uid.isdigit():
            messagebox.showerror("Error", "Enter valid numeric ID.")
            return
        uid = int(uid)
        if os.path.exists("users.csv"):
            df = pd.read_csv("users.csv")
            if uid in df['ID'].values:
                df = df[df['ID'] != uid]
                df.to_csv("users.csv", index=False)
                for f in os.listdir("dataset"):
                    if f.startswith(f"User.{uid}."):
                        os.remove(os.path.join("dataset", f))
                messagebox.showinfo("Deleted", f"Student with ID {uid} deleted.")
            else:
                messagebox.showwarning("Not Found", "Student ID not found.")
        else:
            messagebox.showerror("Error", "No user records found.")

    Button(del_frame, text="Delete Student", command=delete_student).grid(row=1, column=0, columnspan=2, pady=10)

    Label(dash, text="View Student Database", font=("Arial", 12, "bold")).pack(pady=10)
    def view_student_database():
        win = Toplevel()
        win.title("Student Database")
        win.geometry("700x400")
        apply_theme(win)

        table = ttk.Treeview(win, columns=("ID", "Name"), show="headings")
        table.heading("ID", text="ID")
        table.heading("Name", text="Name")
        table.column("ID", width=100)
        table.column("Name", width=300)
        table.pack(fill=BOTH, expand=True)

        if os.path.exists("users.csv"):
            df = pd.read_csv("users.csv")
            students = df[df["Type"] == "student"]
            for _, row in students.iterrows():
                table.insert('', END, values=(row["ID"], row["Name"]))
        else:
            messagebox.showerror("Error", "No student records found.")
    Button(dash, text="View Student Database", command=view_student_database).pack(pady=10)

def show_attendance():
    win = Toplevel()
    win.title("Attendance Records")
    win.geometry("700x400")
    apply_theme(win)

    table = ttk.Treeview(win, columns=("ID", "Name", "DateTime", "Type"), show="headings")
    for col in ("ID", "Name", "DateTime", "Type"):
        table.heading(col, text=col)
        table.column(col, width=150)
    table.pack(fill=BOTH, expand=True)

    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
        for _, row in df.iterrows():
            table.insert('', END, values=list(row))

def main_gui():
    root = Tk()
    root.title("Face Recognition Attendance Pro")
    root.geometry("700x700")
    apply_theme(root)

    Label(root, text="Face Recognition Attendance Pro", font=("Helvetica", 20, "bold")).pack(pady=15)

    frame = Frame(root)
    frame.pack(pady=10)
    apply_theme(frame)

    Label(frame, text="ID:").grid(row=0, column=0, padx=10, pady=5)
    id_entry = Entry(frame)
    id_entry.grid(row=0, column=1)

    Label(frame, text="Name:").grid(row=1, column=0, padx=10, pady=5)
    name_entry = Entry(frame)
    name_entry.grid(row=1, column=1)

    Label(frame, text="Class:").grid(row=2, column=0, padx=10, pady=5)
    class_entry = Entry(frame)
    class_entry.grid(row=2, column=1)

    Label(frame, text="Parent Email:").grid(row=3, column=0, padx=10, pady=5)
    parent_email_entry = Entry(frame)
    parent_email_entry.grid(row=3, column=1)

    Label(frame, text="Student Phone Number:").grid(row=4, column=0, padx=10, pady=5)
    phone_entry = Entry(frame)
    phone_entry.grid(row=4, column=1)

    def register():
        uid = id_entry.get().strip()
        name = name_entry.get().strip()
        student_class = class_entry.get().strip()
        parent_email = parent_email_entry.get().strip()
        student_phone = phone_entry.get().strip()

        if not uid.isdigit() or not name or not student_class or not parent_email or not student_phone:
            messagebox.showerror("Input Error", "All fields must be filled correctly.")
            return

        columns = ['ID', 'Name', 'Class', 'ParentEmail', 'Phone', 'Type']
        df = pd.read_csv("users.csv") if os.path.exists("users.csv") else pd.DataFrame(columns=columns)
        if int(uid) in df['ID'].values:
            messagebox.showwarning("Exists", "ID already exists.")
            return

        new_row = pd.DataFrame([[int(uid), name, student_class, parent_email, student_phone, "student"]], columns=columns)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv("users.csv", index=False)

        capture_faces(int(uid), name)
        messagebox.showinfo("Success", "Student registered and face captured!")

    Button(root, text="Register & Capture", command=register).pack(pady=10)
    Button(root, text="Train Model", command=train_model).pack(pady=10)
    Button(root, text="Student Entry", command=lambda: recognize_and_mark("Entry", "student")).pack(pady=10)
    Button(root, text="Student Exit", command=lambda: recognize_and_mark("Exit", "student")).pack(pady=10)
    Button(root, text="Teacher Login", command=teacher_portal).pack(pady=10)
    Button(root, text="Toggle Theme", command=lambda: toggle_theme(root)).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    print("Launching GUI...")
    main_gui()