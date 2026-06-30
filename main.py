import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
import mysql.connector
import pandas as pd
from datetime import datetime
from tkinter import ttk, messagebox, Label
import signal
import sys

# Configure appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def signal_handler(sig, frame):
    print("\nShutting down gracefully...")
    if 'obj' in globals():
        obj.cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class Face_Recognition_System:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1920x1080+0+0")
        self.root.title("ASFR System")
        self.running = True

        # Database connection
        self.db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="attendance_system",
            port=3306
        )

        # Window management
        self.add_student_window = None
        self.attendance_window = None
        self.attendance_viewer_window = None

        self.create_main_interface()

    def cleanup(self):
        print("Performing cleanup...")
        self.running = False

        # Close all child windows
        if self.add_student_window:
            self.add_student_window.on_close()
        if self.attendance_window:
            self.attendance_window.on_close()
        if self.attendance_viewer_window:
            self.attendance_viewer_window.on_close()

        # Close database connection
        if self.db.is_connected():
            self.db.close()
            print("Database connection closed")

    def create_main_interface(self):
        # Main frame
        self.main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Header
        header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header_frame.pack(pady=20)

        # University logo
        fu_icon = ctk.CTkImage(Image.open(r"images\Fu_icon.png"), size=(150, 150))
        ctk.CTkLabel(header_frame, image=fu_icon, text="").pack(side="left", padx=20)

        # System title
        title_label = ctk.CTkLabel(
            header_frame,
            text="Attendance System Using Facial Recognition",
            font=("Arial Bold", 28),
            text_color="#00B0F0"
        )
        title_label.pack(side="left", padx=20)

        # Main buttons grid
        button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        button_frame.pack(pady=50)

        button_specs = [
            (r"images\Face_Icon.png", "Add Student", self.open_add_student),
            (r"images\train_icon.png", "Train Model", self.open_training),
            (r"images\attendance_mark.png", "Mark Attendance", self.open_attendance),
            (r"images\view_attendance.png", "View Attendance", self.open_attendance_viewer),
            (r"images\attendance_icon.png", "Generate Report", self.open_report)
        ]

        for col, (img_path, text, command) in enumerate(button_specs):
            btn = ctk.CTkButton(
                button_frame,
                text=text,
                image=ctk.CTkImage(Image.open(img_path), size=(80, 80)),
                compound="top",
                command=command,
                width=250,
                height=180,
                corner_radius=15,
                fg_color="#1F538D",
                hover_color="#14375E",
                font=("Arial Bold", 16)
            )
            btn.grid(row=0, column=col, padx=15, pady=10)

    # Window management methods
    def open_add_student(self):
        if self.add_student_window and self.add_student_window.window.winfo_exists():
            self.add_student_window.window.lift()
            return
        self.add_student_window = AddStudentWindow(self)

    def open_attendance(self):
        if self.attendance_window and self.attendance_window.window.winfo_exists():
            self.attendance_window.window.lift()
            return
        self.attendance_window = AttendanceWindow(self)

    def open_attendance_viewer(self):
        if self.attendance_viewer_window and self.attendance_viewer_window.window.winfo_exists():
            self.attendance_viewer_window.window.lift()
            return
        self.attendance_viewer_window = AttendanceViewerWindow(self)

    def open_training(self):
        self.train_model()

    def open_report(self):
        self.generate_report()

    # Core functionality methods
    def train_model(self):
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            faces = []
            ids = []

            dataset_path = "dataset"
            image_files = [f for f in os.listdir(dataset_path) if f.startswith("User_")]

            for file_name in image_files:
                try:
                    parts = file_name.split("_")
                    student_id = parts[1]
                    file_path = os.path.join(dataset_path, file_name)
                    img = Image.open(file_path).convert('L')
                    img_np = np.array(img, 'uint8')
                    faces.append(img_np)
                    ids.append(int(student_id))
                except Exception as e:
                    continue

            if len(faces) == 0:
                messagebox.showerror("Error", "No valid faces found in dataset", parent=self.root)
                return

            recognizer.train(faces, np.array(ids))
            recognizer.save("trained_model.yml")
            messagebox.showinfo("Success", f"Model trained with {len(faces)} samples!", parent=self.root)
        except Exception as e:
            messagebox.showerror("Training Error", str(e), parent=self.root)

    def generate_report(self):
        try:
            query = """
            SELECT students.id, students.name, attendance.date, attendance.time 
            FROM students 
            JOIN attendance ON students.id = attendance.id
            """
            df = pd.read_sql(query, self.db)
            df.to_csv("attendance_report.csv", index=False)
            messagebox.showinfo("Success", "Report saved as attendance_report.csv", parent=self.root)
        except Exception as e:
            messagebox.showerror("Report Error", str(e), parent=self.root)


class AddStudentWindow:
    def __init__(self, parent):
        self.parent = parent
        self.window = ctk.CTkToplevel(parent.root)
        self.window.title("Student Registration")
        self.window.geometry("1280x720")
        self.window.transient(parent.root)
        self.window.grab_set()

        self.capturing = False
        self.count = 0
        self.student_id = ""
        self.name = ""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cam = None

        try:
            self.cam = cv2.VideoCapture(2)
            if not self.cam.isOpened():
                raise Exception("Camera not available")
            self.create_interface()
            self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        except Exception as e:
            messagebox.showerror("Camera Error", str(e), parent=self.parent.root)
            self.window.destroy()

    def create_interface(self):
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Form section
        form_frame = ctk.CTkFrame(main_frame, width=400, corner_radius=15)
        form_frame.pack(side="left", padx=20, pady=20, fill="y")

        ctk.CTkLabel(form_frame, text="Student Registration",
                     font=("Arial Bold", 20)).pack(pady=15)

        self.id_entry = ctk.CTkEntry(
            form_frame,
            placeholder_text="Student ID",
            width=300,
            height=40,
            corner_radius=10
        )
        self.id_entry.pack(pady=10)

        self.name_entry = ctk.CTkEntry(
            form_frame,
            placeholder_text="Student Name",
            width=300,
            height=40,
            corner_radius=10
        )
        self.name_entry.pack(pady=10)

        ctk.CTkButton(
            form_frame,
            text="Start Capture",
            command=self.start_capture,
            width=200,
            height=40,
            corner_radius=10,
            fg_color="#1F538D",
            hover_color="#14375E"
        ).pack(pady=20)

        # Video feed section
        video_frame = ctk.CTkFrame(main_frame, corner_radius=15)
        video_frame.pack(side="left", padx=20, pady=20, fill="both", expand=True)

        self.video_label = Label(video_frame)
        self.video_label.pack(padx=10, pady=10, fill="both", expand=True)
        self.update_video_feed()

        # Delete section
        delete_frame = ctk.CTkFrame(main_frame, corner_radius=15)
        delete_frame.pack(side="right", padx=20, pady=20, fill="y")

        ctk.CTkLabel(delete_frame, text="Delete Student",
                     font=("Arial Bold", 16)).pack(pady=10)

        self.selected_student = ctk.StringVar()
        self.student_dropdown = ctk.CTkComboBox(
            delete_frame,
            variable=self.selected_student,
            values=["Select Student"],
            state="readonly",
            width=200,
            dropdown_fg_color="#2B2B2B",
            button_color="#1F538D"
        )
        self.student_dropdown.pack(pady=10)

        ctk.CTkButton(
            delete_frame,
            text="Delete Student",
            command=self.delete_student,
            width=150,
            fg_color="#B22222",
            hover_color="#8B0000"
        ).pack(pady=10)

        self.load_students()

    def load_students(self):
        cursor = self.parent.db.cursor()
        cursor.execute("SELECT id FROM students")
        students = [str(row[0]) for row in cursor.fetchall()]
        self.student_dropdown.configure(values=["Select Student"] + students)
        self.selected_student.set("Select Student")

    def delete_student(self):
        selected_id = self.selected_student.get()
        if selected_id == "Select Student" or not selected_id:
            messagebox.showwarning("Warning", "Please select a student to delete", parent=self.window)
            return

        try:
            cursor = self.parent.db.cursor()
            cursor.execute("DELETE FROM students WHERE id = %s", (selected_id,))
            self.parent.db.commit()

            # Delete associated face images
            for f in os.listdir("dataset"):
                if f.startswith(f"User_{selected_id}_"):
                    os.remove(os.path.join("dataset", f))

            messagebox.showinfo("Success", "Student deleted successfully", parent=self.window)
            self.load_students()
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self.window)

    def update_video_feed(self):
        if not self.parent.running or not self.window.winfo_exists():
            return

        if self.capturing and self.count >= 100:
            self.capturing = False
            messagebox.showinfo("Success",
                                f"Captured {self.count} images for {self.name}", parent=self.window)

        ret, frame = self.cam.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                if self.capturing and self.count < 100:
                    self.count += 1
                    cv2.imwrite(f"dataset/User_{self.student_id}_{self.count}.jpg", gray[y:y + h, x:x + w])

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.window.after(10, self.update_video_feed)

    def start_capture(self):
        self.student_id = self.id_entry.get()
        self.name = self.name_entry.get()

        if not self.student_id or not self.name:
            messagebox.showerror("Error", "Please fill all fields", parent=self.window)
            return

        try:
            cursor = self.parent.db.cursor()
            cursor.execute("INSERT INTO students VALUES (%s, %s)", (self.student_id, self.name))
            self.parent.db.commit()
        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", str(err), parent=self.window)
            return

        self.capturing = True
        self.count = 0
        if not os.path.exists("dataset"):
            os.makedirs("dataset")

    def on_close(self):
        if self.cam and self.cam.isOpened():
            self.cam.release()
        self.parent.add_student_window = None
        self.window.destroy()


class AttendanceWindow:
    def __init__(self, parent):
        self.parent = parent
        self.window = ctk.CTkToplevel(parent.root)
        self.window.title("Attendance Marking")
        self.window.geometry("1280x720")
        self.window.transient(parent.root)
        self.window.grab_set()

        self.recognized_ids = set()
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("trained_model.yml")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cam = None

        try:
            self.cam = cv2.VideoCapture(2)
            if not self.cam.isOpened():
                raise Exception("Camera not available")
            self.create_interface()
            self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        except Exception as e:
            messagebox.showerror("Camera Error", str(e), parent=self.parent.root)
            self.window.destroy()

    def create_interface(self):
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        # Status label
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Ready to mark attendance",
            font=("Arial Bold", 16),
            text_color="#00FF00"
        )
        self.status_label.pack(pady=20)

        # Video feed section
        video_frame = ctk.CTkFrame(main_frame, corner_radius=15)
        video_frame.pack(pady=20, fill="both", expand=True)

        self.video_label = Label(video_frame)
        self.video_label.pack(padx=10, pady=10, fill="both", expand=True)

        self.update_video_feed()



    def update_video_feed(self):
        if not self.parent.running or not self.window.winfo_exists():
            return

        ret, frame = self.cam.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])

                if confidence < 47:
                    cursor = self.parent.db.cursor()
                    cursor.execute("SELECT name FROM students WHERE id=%s", (id,))
                    result = cursor.fetchone()

                    if result:
                        name = result[0]
                        cv2.putText(frame, name, (x + 5, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        if id not in self.recognized_ids:
                            try:
                                cursor.execute(
                                    "INSERT INTO attendance (id, date, time) VALUES (%s, CURDATE(), CURTIME())",
                                    (id,))
                                self.parent.db.commit()
                                self.recognized_ids.add(id)
                                self.status_label.configure(
                                    text=f"Marked: {name} ({datetime.now().strftime('%H:%M:%S')})",
                                    text_color="green"
                                )
                            except Exception as e:
                                self.status_label.configure(
                                    text=f"Error: {str(e)}",
                                    text_color="red"
                                )
                else:
                    cv2.putText(frame, "Unknown", (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.window.after(10, self.update_video_feed)

    def on_close(self):
        if self.cam and self.cam.isOpened():
            self.cam.release()
        self.parent.attendance_window = None
        self.window.destroy()


class AttendanceViewerWindow:
    def __init__(self, parent):
        self.parent = parent
        self.window = ctk.CTkToplevel(parent.root)
        self.window.title("Attendance Viewer")
        self.window.geometry("1280x720")
        self.window.transient(parent.root)
        self.window.grab_set()

        self.create_interface()
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_interface(self):
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Treeview container
        tree_frame = ctk.CTkFrame(main_frame, corner_radius=15)
        tree_frame.pack(fill="both", expand=True, pady=20)

        # Treeview styling
        style = ttk.Style()
        style.theme_use("default")

        style.configure("Treeview",
                        background="#2a2d2e",
                        foreground="white",
                        fieldbackground="#2a2d2e",
                        font=('Arial', 12),
                        rowheight=25,
                        borderwidth=0
                        )
        style.configure("Treeview.Heading",
                        background="#3b3b3b",
                        foreground="white",
                        font=('Arial Bold', 14),
                        relief="flat"
                        )
        style.map("Treeview",
                  background=[('selected', '#22559b')],
                  foreground=[('selected', 'white')]
                  )

        columns = ("ID", "Name", "Date", "Time", "Status")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings")

        # Configure columns
        col_widths = [45, 200, 150, 150, 100]
        for col, width in zip(columns, col_widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width, anchor="center")

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Grid layout
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        # Control buttons
        btn_frame = ctk.CTkFrame(main_frame)
        btn_frame.pack(pady=10)

        ctk.CTkButton(
            btn_frame,
            text="Delete Selected Record",
            command=self.delete_record,
            fg_color="#B22222",
            hover_color="#8B0000"
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            btn_frame,
            text="Refresh List",
            command=self.load_data,
            fg_color="#1F538D",
            hover_color="#14375E"
        ).pack(side="left", padx=10)

        self.load_data()

    def load_data(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        try:
            cursor = self.parent.db.cursor()
            cursor.execute("""
                SELECT students.id, students.name, attendance.date, attendance.time 
                FROM students 
                JOIN attendance ON students.id = attendance.id
                ORDER BY attendance.date DESC, attendance.time DESC
            """)
            rows = cursor.fetchall()

            for row in rows:
                self.tree.insert("", "end", values=(
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    "Present"
                ))

            if not rows:
                messagebox.showinfo("Info", "No attendance records found", parent=self.window)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}", parent=self.window)

    def delete_record(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a record to delete", parent=self.window)
            return

        item_data = self.tree.item(selected_item[0], 'values')
        try:
            cursor = self.parent.db.cursor()
            cursor.execute("""
                DELETE FROM attendance 
                WHERE id = %s AND date = %s AND time = %s
            """, (item_data[0], item_data[2], item_data[3]))
            self.parent.db.commit()
            messagebox.showinfo("Success", "Attendance record deleted successfully", parent=self.window)
            self.load_data()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete record: {str(e)}", parent=self.window)

    def on_close(self):
        self.parent.attendance_viewer_window = None
        self.window.destroy()


if __name__ == "__main__":
    root = ctk.CTk()
    obj = None
    try:
        obj = Face_Recognition_System(root)
        root.mainloop()
    except KeyboardInterrupt:
        print("Application closed by user")
    finally:
        if obj:
            obj.cleanup()
        sys.exit(0)