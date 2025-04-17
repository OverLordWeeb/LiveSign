import cv2
import torch
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import torchvision.transforms as transforms
import torch.nn as nn
import time


class SignLanguageModel(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Sign Language Translator")
        self.root.geometry("1280x720")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1, minsize=300)

        self.camera_frame = tk.Frame(self.root, bg="black")
        self.camera_frame.grid(row=0, column=0, sticky="nsew")

        self.text_frame = tk.Frame(self.root)
        self.text_frame.grid(row=0, column=1, sticky="nsew")

        self.label = tk.Label(self.camera_frame)
        self.label.pack(expand=True)

        self.text_display = scrolledtext.ScrolledText(self.text_frame, wrap=tk.WORD)
        self.text_display.pack(fill=tk.BOTH, expand=True)
        self.log_message("Sign Language Translator Initialized...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.classes = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 
            'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Z',
            'Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'
        ]

        self.model = SignLanguageModel(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(torch.load(r"C:\Users\pkucz\Desktop\SignLanguage\sign_language_model.pth", map_location=self.device))
        self.model.eval()

        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            self.log_message("Error: Could not open webcam.")
            self.root.destroy()
            return

        self.last_prediction_time = 0
        self.prediction_delay = 2000  # milliseconds
        self.latest_prediction = ("", 0.0)

        self.update_frame()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def log_message(self, message):
        self.text_display.insert(tk.END, message + "\n")
        self.text_display.see(tk.END)

    def predict_sign(self, frame):
        img = Image.fromarray(frame)
        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
            prediction_label = self.classes[predicted.item()]
            confidence_percent = confidence.item() * 100

        return prediction_label, confidence_percent

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Draw green focus box
            h, w, _ = display_frame.shape
            box_size = 200
            top_left = (w // 2 - box_size // 2, h // 2 - box_size // 2)
            bottom_right = (w // 2 + box_size // 2, h // 2 + box_size // 2)
            cv2.rectangle(display_frame, top_left, bottom_right, (0, 255, 0), 2)

            # Predict every few seconds
            current_time = int(time.time() * 1000)
            if current_time - self.last_prediction_time >= self.prediction_delay:
                focus_area = display_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                resized_for_model = cv2.resize(focus_area, (64, 64))
                prediction, confidence = self.predict_sign(resized_for_model)
                self.latest_prediction = (prediction, confidence)
                self.log_message(f"Detected Sign: {prediction} ({confidence:.2f}%)")
                self.last_prediction_time = current_time

            # Show prediction on green box
            if self.latest_prediction[0]:
                pred_text = f"{self.latest_prediction[0]} ({self.latest_prediction[1]:.1f}%)"
                text_pos = (top_left[0], top_left[1] - 10)
                cv2.putText(display_frame, pred_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Convert for Tkinter display
            img = Image.fromarray(display_frame)
            frame_width = self.camera_frame.winfo_width()
            frame_height = self.camera_frame.winfo_height()
            if frame_width > 1 and frame_height > 1:
                img_resized = img.resize((frame_width, frame_height), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img_resized)
                self.label.imgtk = imgtk
                self.label.config(image=imgtk)

        self.root.after(15, self.update_frame)

    def on_closing(self):
        self.log_message("Closing application...")
        if self.video_capture.isOpened():
            self.video_capture.release()
        cv2.destroyAllWindows()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
