import os
import sqlite3
from datetime import datetime
import cv2
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox, filedialog
from tkinter.simpledialog import askstring
import pandas as pd

# Set up database paths
db_path = "attendance_system.db"
students_data_path = "students_data"

# Ensure OpenCV's contrib package is installed
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    messagebox.showerror("Error", "Please install 'opencv-contrib-python' to enable face recognition.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create database and tables if they don't exist


def create_database():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS Students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT UNIQUE,
                    name TEXT,
                    section TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS Attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    name TEXT,
                    section TEXT,
                    date TEXT,
                    time TEXT
                )''')
    conn.commit()
    conn.close()


create_database()

# Ensure student data directory exists
if not os.path.exists(students_data_path):
    os.makedirs(students_data_path)

# Train the recognizer with existing face data


def train_model():
    faces = []
    ids = []
    for student_id in os.listdir(students_data_path):
        if not student_id.isdigit():
            continue  # Skip non-numeric folders
        folder_path = os.path.join(students_data_path, student_id)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if gray_image is not None:
                    faces.append(gray_image)
                    ids.append(int(student_id))

    if faces and ids:
        recognizer.train(faces, np.array(ids))
    else:
        print("No data to train the recognizer.")

# Helper functions


def execute_sql(query, params=(), fetch=False):
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute(query, params)
        if fetch:
            return c.fetchall()
        conn.commit()
    except sqlite3.Error as e:
        messagebox.showerror("Database Error", f"An error occurred: {e}")
    finally:
        conn.close()


def get_students():
    return execute_sql("SELECT * FROM Students", fetch=True)


def add_student(student_id, name, section):
    try:
        if not student_id.isdigit():
            raise ValueError("Student ID must be numeric.")
        execute_sql("INSERT INTO Students (student_id, name, section) VALUES (?, ?, ?)", (student_id, name, section))
    except sqlite3.IntegrityError:
        messagebox.showerror("Error", "Student ID already exists.")
    except ValueError as ve:
        messagebox.showerror("Error", str(ve))


def log_attendance(student_id, name, section):
    date = datetime.now().strftime("%Y-%m-%d")
    # Check if the student has already been marked present today
    existing_attendance = execute_sql("SELECT * FROM Attendance WHERE student_id = ? AND date = ?",
                                      (student_id, date), fetch=True)
    if not existing_attendance:
        time = datetime.now().strftime("%H:%M:%S")
        execute_sql("INSERT INTO Attendance (student_id, name, section, date, time) VALUES (?, ?, ?, ?, ?)",
                    (student_id, name, section, date, time))


def export_attendance_to_csv():
    data = execute_sql("SELECT * FROM Attendance", fetch=True)
    if not data:
        messagebox.showinfo("Export Error", "No attendance data available for export.")
        return
    df = pd.DataFrame(data, columns=["ID", "Student ID", "Name", "Section", "Date", "Time"])
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file_path:
        df.to_csv(file_path, index=False)
        messagebox.showinfo("Export Successful", "Attendance data exported successfully!")

# GUI: Tkinter Application


class AttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Attendance System")
        self.root.geometry("800x600")

        self.video_capture = None
        self.is_recording = False

        # Main Menu
        self.main_menu = tk.Frame(root)
        self.main_menu.pack(fill=tk.BOTH, expand=True)

        tk.Label(self.main_menu, text="Attendance System", font=("Arial", 20)).pack(pady=20)
        tk.Button(self.main_menu, text="Start Attendance", font=("Arial", 14), command=self.start_attendance).pack(pady=10)
        tk.Button(self.main_menu, text="View Attendance Log", font=("Arial", 14), command=self.view_attendance_log).pack(pady=10)
        tk.Button(self.main_menu, text="Register New Student", font=("Arial", 14), command=self.register_student).pack(pady=10)
        tk.Button(self.main_menu, text="Exit", font=("Arial", 14), command=self.root.quit).pack(pady=10)

    def start_attendance(self):
        train_model()  # Ensure the model is trained before starting recognition
        if not self.is_recording:
            self.is_recording = True
            self.video_capture = cv2.VideoCapture(0)
            self.recognize_faces()
        else:
            self.is_recording = False
            if self.video_capture:
                self.video_capture.release()
            cv2.destroyAllWindows()

    def recognize_faces(self):
        def process_frame():
            ret, frame = self.video_capture.read()
            if not ret:
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                try:
                    student_id, confidence = recognizer.predict(face)
                    if confidence < 50:
                        student = [row for row in get_students() if row[1] == str(student_id)]
                        if student:
                            name, section = student[0][2], student[0][3]
                            log_attendance(student_id, name, section)
                            cv2.putText(frame, f"{name} ({section})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                except cv2.error:
                    cv2.putText(frame, "No Model Data", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow("Face Recognition", frame)

            if self.is_recording:
                self.root.after(10, process_frame)

        process_frame()

    def register_student(self):
        def capture_images(student_id):
            cap = cv2.VideoCapture(0)
            count = 0
            while count < 20:  # Capture 20 images
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                for (x, y, w, h) in faces:
                    count += 1
                    face = gray[y:y+h, x:x+w]
                    folder = os.path.join(students_data_path, student_id)
                    os.makedirs(folder, exist_ok=True)
                    file_path = os.path.join(folder, f"{count}.jpg")
                    cv2.imwrite(file_path, face)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                cv2.imshow("Capturing Images", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        def save_student():
            student_id = student_id_entry.get()
            name = name_entry.get()
            section = section_entry.get()
            if not student_id or not name or not section:
                messagebox.showerror("Error", "All fields are required!")
                return

            password = askstring("Password", "Enter password to register new student:")
            if password != "adminpassword":  # Update with your password
                messagebox.showerror("Error", "Incorrect password!")
                return

            add_student(student_id, name, section)
            capture_images(student_id)
            train_model()  # Train the model after adding new data
            messagebox.showinfo("Success", "Student registered successfully!")
            register_window.destroy()

        register_window = tk.Toplevel(self.root)
        register_window.title("Register New Student")
        register_window.geometry("400x300")

        tk.Label(register_window, text="Student ID").pack()
        student_id_entry = tk.Entry(register_window)
        student_id_entry.pack()

        tk.Label(register_window, text="Name").pack()
        name_entry = tk.Entry(register_window)
        name_entry.pack()

        tk.Label(register_window, text="Section").pack()
        section_entry = tk.Entry(register_window)
        section_entry.pack()

        tk.Button(register_window, text="Register", command=save_student).pack(pady=10)

    def view_attendance_log(self):
        log_window = tk.Toplevel(self.root)
        log_window.title("Attendance Log")
        log_window.geometry("700x500")

        frame = tk.Frame(log_window)
        frame.pack(fill=tk.BOTH, expand=True)

        tree = ttk.Treeview(frame, columns=("Student ID", "Name", "Section", "Date", "Time"), show="headings")
        tree.heading("Student ID", text="Student ID")
        tree.heading("Name", text="Name")
        tree.heading("Section", text="Section")
        tree.heading("Date", text="Date")
        tree.heading("Time", text="Time")

        tree.column("Student ID", width=100, anchor=tk.CENTER)
        tree.column("Name", width=150, anchor=tk.W)
        tree.column("Section", width=100, anchor=tk.CENTER)
        tree.column("Date", width=100, anchor=tk.CENTER)
        tree.column("Time", width=100, anchor=tk.CENTER)

        scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        attendance_data = execute_sql("SELECT * FROM Attendance", fetch=True)
        if not attendance_data:
            messagebox.showinfo("No Data", "No attendance records available.")
            return

        for record in attendance_data:
            tree.insert("", tk.END, values=(record[1], record[2], record[3], record[4], record[5]))

        export_button = tk.Button(log_window, text="Export to CSV", command=export_attendance_to_csv)
        export_button.pack(pady=10)

# Run the application


root = tk.Tk()
app = AttendanceSystem(root)
root.mainloop()
