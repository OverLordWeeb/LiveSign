import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import time
import mediapipe as mp

# Sign Language Model definition
class SignLanguageModel(nn.Module):
    def __init__(self, num_classes):
        super(SignLanguageModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
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

# Main application
class HandSignApp:
    def __init__(self, window, model_path, classes):
        self.window = window
        self.window.title("Hand Tracker + Sign Recognition")
        self.window.geometry("900x600")

        self.classes = classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SignLanguageModel(len(self.classes)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.cap = cv2.VideoCapture(0)

        self.label = Label(self.window)
        self.label.pack()

        self.pred_label = Label(self.window, text="Prediction: None", font=("Arial", 24))
        self.pred_label.pack()

        self.last_pred_time = 0
        self.pred_delay = 2

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        self.update_frame()

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
            confidence, predicted = torch.max(probs, 1)
            pred_class = self.classes[predicted.item()]
            return pred_class, confidence.item()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Extract bounding box for hand ROI
                    landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    x_min = int(np.min(landmark_array[:, 0]) * frame.shape[1])
                    y_min = int(np.min(landmark_array[:, 1]) * frame.shape[0])
                    x_max = int(np.max(landmark_array[:, 0]) * frame.shape[1])
                    y_max = int(np.max(landmark_array[:, 1]) * frame.shape[0])
                    hand_roi = frame[y_min:y_max, x_min:x_max]

                    if hand_roi is not None and (time.time() - self.last_pred_time) >= self.pred_delay:
                        pred_class, confidence = self.predict(hand_roi)
                        self.pred_label.config(text=f"Prediction: {pred_class} ({confidence*100:.2f}%)")
                        self.last_pred_time = time.time()

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.config(image=imgtk)

        self.window.after(10, self.update_frame)

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.window.destroy()


if __name__ == "__main__":
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Z', 'Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

    root = Tk()
    app = HandSignApp(root, r"C:\Users\pkucz\Desktop\SignLanguage\sign_language_model.pth", classes)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
