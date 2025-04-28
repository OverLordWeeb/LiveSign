import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import time
import mediapipe as mp
import json
from typing import Any

class SignLanguageModel(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class SignTranslatorApp:
    def __init__(self, window, model_path):
        self.window = window
        self.window.title("SignLive Chat Translator")
        self.window.geometry("1200x700")

        with open("class_labels.json", "r") as f:
            self.classes = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SignLanguageModel(len(self.classes)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.video_frame = Label(self.window)
        self.video_frame.pack(side=LEFT, fill=BOTH, expand=True)

        right_frame = Frame(self.window)
        right_frame.pack(side=RIGHT, fill=BOTH)

        self.chat_box = Text(right_frame, width=40, font=("Arial", 16))
        self.chat_box.pack(fill=BOTH, expand=True)
        self.chat_box.insert(END, "Signed Text:\n")

        button_frame = Frame(right_frame)
        button_frame.pack(fill=X)

        save_button = Button(button_frame, text="Save Text", command=self.save_text)
        save_button.pack(side=LEFT, padx=5, pady=5)

        load_button = Button(button_frame, text="Load Text", command=self.load_text)
        load_button.pack(side=LEFT, padx=5, pady=5)

        clear_button = Button(button_frame, text="Clear Text", command=self.clear_text)
        clear_button.pack(side=LEFT, padx=5, pady=5)

        self.paused = False
        self.pause_button = Button(button_frame, text="Pause/Resume", command=self.toggle_pause)
        self.pause_button.pack(side=LEFT, padx=5, pady=5)

        threshold_frame = Frame(right_frame)
        threshold_frame.pack(fill=X)
        Label(threshold_frame, text="Confidence Threshold:").pack(side=LEFT, padx=5)
        self.confidence_threshold = Scale(threshold_frame, from_=0, to=100, orient=HORIZONTAL)
        self.confidence_threshold.set(50)
        self.confidence_threshold.pack(fill=X, padx=5)

        self.prediction_buffer = ""
        self.last_pred = ""
        self.last_pred_time = 0
        self.last_hand_time = time.time()
        self.space_timeout = 2.5
        self.pred_delay = 1.5
        self.pred_history = []
        self.hand_still_start_time = None
        self.last_bbox = None

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        self.update_frame()

    def save_text(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if filepath:
            with open(filepath, "w") as f:
                f.write(self.chat_box.get("1.0", END))

    def load_text(self):
        filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if filepath:
            with open(filepath, "r") as f:
                text = f.read()
                self.chat_box.delete("1.0", END)
                self.chat_box.insert(END, text)

    def clear_text(self):
        self.chat_box.delete("1.0", END)

    def toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.pause_button.config(bg="red")
        else:
            self.pause_button.config(bg="SystemButtonFace")

    def preprocess(self, roi):
        roi = cv2.resize(roi, (64, 64))
        img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        return img_tensor

    def predict(self, roi):
        img_tensor = self.preprocess(roi)
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            top_probs, top_idxs = torch.topk(probs, 2)
            pred_class = self.classes[top_idxs[0][0].item()]
            self.pred_history.append(pred_class)
            if len(self.pred_history) > 5:
                self.pred_history.pop(0)
            most_common = max(set(self.pred_history), key=self.pred_history.count)
            pred_agreement = self.pred_history.count(most_common) / len(self.pred_history)
            if pred_agreement >= 0.6:
                return [(most_common, top_probs[0][0].item())]
            else:
                return [(self.last_pred, 0.0)]

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results: Any = self.hands.process(rgb_frame)

        h, w, _ = frame.shape
        all_points = []
        text_1, text_2 = "No hands", ""

        if results.multi_hand_landmarks:
            self.last_hand_time = time.time()
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    all_points.append([lm.x, lm.y])

            all_points = np.array(all_points)
            x_min = max(int(np.min(all_points[:, 0]) * w) - 20, 0)
            y_min = max(int(np.min(all_points[:, 1]) * h) - 20, 0)
            x_max = min(int(np.max(all_points[:, 0]) * w) + 20, w)
            y_max = min(int(np.max(all_points[:, 1]) * h) + 20, h)

            roi = frame[y_min:y_max, x_min:x_max]
            current_bbox = (x_min, y_min, x_max, y_max)

            if self.last_bbox is not None:
                diff = np.sum(np.abs(np.array(current_bbox) - np.array(self.last_bbox)))
                if diff < 20:
                    if self.hand_still_start_time is None:
                        self.hand_still_start_time = time.time()
                else:
                    self.hand_still_start_time = None
            self.last_bbox = current_bbox

            if roi.size > 0 and not self.paused and (time.time() - self.last_pred_time) > self.pred_delay:
                if self.hand_still_start_time and (time.time() - self.hand_still_start_time) >= 1.0:
                    predictions = self.predict(roi)
                    top_pred, confidence = predictions[0]
                    threshold = self.confidence_threshold.get() / 100.0
                    if confidence > threshold and top_pred != self.last_pred:
                        self.prediction_buffer += top_pred
                        self.chat_box.insert(END, top_pred)
                        self.last_pred = top_pred
                        self.last_pred_time = time.time()

                    text_1 = f"Prediction: {top_pred} ({confidence*100:.1f}%)"

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        elif time.time() - self.last_hand_time > self.space_timeout:
            self.last_hand_time = time.time()
            self.prediction_buffer += " "
            self.chat_box.insert(END, " ")

        cv2.putText(frame, text_1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.config(image=imgtk)

        self.window.after(15, self.update_frame)

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.window.destroy()

if __name__ == "__main__":
    root = Tk()
    app = SignTranslatorApp(root, "sign_language_model.pth")
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
