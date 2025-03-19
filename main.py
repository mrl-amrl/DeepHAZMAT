import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
from imutils import resize
from deep_hazmat import DeepHAZMAT
import json
import os

class HazmatDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hazmat Detector")
        self.root.geometry("1000x650")
        self.root.configure(bg="#2B2B2B")

        # Load hazmat data
        self.hazmat_data = self.load_hazmat_data()
        self.detail_window = None

        # Style configuration
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 12), padding=10, background="#4A90E2", foreground="#333333")
        style.map("TButton", background=[("active", "#357ABD")], foreground=[("active", "#222222")])

        # Main layout: Sidebar and Canvas
        self.sidebar = tk.Frame(root, bg="#3C3C3C", width=200)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.main_frame = tk.Frame(root, bg="#2B2B2B")
        self.main_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Sidebar widgets
        tk.Label(self.sidebar, text="Hazmat Detector", font=("Helvetica", 16, "bold"), fg="white", bg="#3C3C3C").pack(pady=20)

        self.select_image_button = ttk.Button(self.sidebar, text="Select Image", command=self.load_image)
        self.select_image_button.pack(fill="x", pady=5)

        self.select_video_button = ttk.Button(self.sidebar, text="Select Video", command=self.load_video)
        self.select_video_button.pack(fill="x", pady=5)

        self.run_button = ttk.Button(self.sidebar, text="Run Detection", command=self.run_detection, state="disabled")
        self.run_button.pack(fill="x", pady=5)

        # Canvas with border
        self.canvas_frame = tk.Frame(self.main_frame, bg="#FFFFFF", bd=2, relief="solid")
        self.canvas_frame.pack(fill="both", expand=True)
        self.canvas = tk.Label(self.canvas_frame, bg="#000000")
        self.canvas.pack(fill="both", expand=True)

        self.file_path = None
        self.is_video = False
        self.video_capture = None
        self.running_video = False

        self.deep_hazmat = DeepHAZMAT(k=0, net_directory="net", min_confidence=0.8, nms_threshold=0.3, segmentation_enabled=True)

    def load_hazmat_data(self):
        data_path = os.path.join("data", "hazmat_data.json")
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: {data_path} not found. No hazmat data loaded.")
            return {}

    def load_image(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if self.file_path:
            print(f"Selected image: {self.file_path}")  # Debug
            self.is_video = False
            self.display_image(self.file_path)
            self.run_button.state(["!disabled"])

    def load_video(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if self.file_path:
            print(f"Selected video: {self.file_path}")  # Debug
            self.is_video = True
            self.run_button.state(["!disabled"])
        else:
            print("No video selected")  # Debug

    def display_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image at {image_path}")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((850, 550))
        img_tk = ImageTk.PhotoImage(image)
        self.canvas.config(image=img_tk)
        self.canvas.image = img_tk

    def run_detection(self):
        if not self.file_path:
            print("No file path set")  # Debug
            return
        print(f"Running detection on: {self.file_path}, is_video: {self.is_video}")  # Debug
        if self.is_video:
            self.process_video()
        else:
            self.process_image()

    def process_image(self):
        image = cv2.imread(self.file_path)
        if image is None:
            print(f"Error: Could not load image at {self.file_path}")
            return
        image = resize(image, width=850)

        detections = self.deep_hazmat.update(image)
        for hazmat in detections:
            hazmat.draw(image=image, padding=0.1)
            self.show_detection_details(hazmat)

        self.show_result(image)

    def process_video(self):
        if self.video_capture:
            self.video_capture.release()

        self.video_capture = cv2.VideoCapture(self.file_path)
        if not self.video_capture.isOpened():
            print(f"Error: Could not open video at {self.file_path}")
            return
        print(f"Video opened successfully: {self.file_path}")  # Debug
        self.running_video = True
        self.update_video_frame()

    def update_video_frame(self):
        if not self.running_video:
            return

        ret, frame = self.video_capture.read()
        if not ret:
            print("End of video or error reading frame")  # Debug
            self.running_video = False
            self.video_capture.release()
            return

        frame = resize(frame, width=850)
        detections = self.deep_hazmat.update(frame)
        if detections:
            self.show_detection_details(detections[0])
        for hazmat in detections:
            hazmat.draw(image=frame, padding=0.1)

        self.show_result(frame)
        self.root.after(20, self.update_video_frame)

    def show_result(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((850, 550))
        img_tk = ImageTk.PhotoImage(image)
        self.canvas.config(image=img_tk)
        self.canvas.image = img_tk

    def show_detection_details(self, hazmat):
        label = hazmat.name.lower()
        if label in self.hazmat_data:
            details = self.hazmat_data[label]
        else:
            details = {
                "description": "No data available.",
                "explosiveness": "Unknown",
                "intensity": "Unknown",
                "reaction_to_elements": "Unknown",
                "storage": "Unknown",
                "waste_disposal": "Unknown"
            }

        if self.detail_window is None or not self.detail_window.winfo_exists():
            self.detail_window = tk.Toplevel(self.root)
            self.detail_window.title("Detection Details")
            self.detail_window.geometry("450x350")
            self.detail_window.configure(bg="#F5F6F5")
            self.detail_window.resizable(True, True)

            canvas = tk.Canvas(self.detail_window, bg="#F5F6F5")
            scrollbar_y = ttk.Scrollbar(self.detail_window, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas, bg="white", bd=1, relief="solid")

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar_y.set)

            canvas.pack(side="left", fill="both", expand=True, padx=20, pady=20)
            scrollbar_y.pack(side="right", fill="y")

            tk.Label(scrollable_frame, text=label.capitalize(), font=("Helvetica", 16, "bold"), bg="white", fg="#333333").pack(pady=(10, 5))
            for key, value in details.items():
                tk.Label(scrollable_frame, text=f"{key.capitalize().replace('_', ' ')}:", font=("Helvetica", 10, "bold"), bg="white", fg="#555555").pack(anchor="w", padx=10)
                tk.Label(scrollable_frame, text=value, font=("Helvetica", 10), bg="white", fg="#666666", wraplength=380, justify="left").pack(anchor="w", padx=20, pady=(0, 10))
        else:
            for widget in self.detail_window.winfo_children():
                widget.destroy()

            canvas = tk.Canvas(self.detail_window, bg="#F5F6F5")
            scrollbar_y = ttk.Scrollbar(self.detail_window, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas, bg="white", bd=1, relief="solid")

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar_y.set)

            canvas.pack(side="left", fill="both", expand=True, padx=20, pady=20)
            scrollbar_y.pack(side="right", fill="y")

            tk.Label(scrollable_frame, text=label.capitalize(), font=("Helvetica", 16, "bold"), bg="white", fg="#333333").pack(pady=(10, 5))
            for key, value in details.items():
                tk.Label(scrollable_frame, text=f"{key.capitalize().replace('_', ' ')}:", font=("Helvetica", 10, "bold"), bg="white", fg="#555555").pack(anchor="w", padx=10)
                tk.Label(scrollable_frame, text=value, font=("Helvetica", 10), bg="white", fg="#666666", wraplength=380, justify="left").pack(anchor="w", padx=20, pady=(0, 10))

if __name__ == "__main__":
    root = tk.Tk()
    app = HazmatDetectorApp(root)
    root.mainloop()
