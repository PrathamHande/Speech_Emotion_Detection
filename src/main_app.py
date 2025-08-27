import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import soundfile as sf
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import threading

# --- Global Configuration ---
MODELS_PATH = os.path.join("src", "models")
EMOTIONS = ['calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
GENDERS = ['male', 'female']
N_MFCC = 40
SAMPLE_RATE = 44100  # Standard audio sample rate


class EmotionDetectionApp:
    def __init__(self, root):
        """Initializes the main application window and loads models."""
        self.root = root
        self.root.title("Voice Emotion Detection")
        self.root.geometry("600x400")
        self.root.configure(bg="#2c3e50")

        # --- Load Models ---
        self.load_models()

        # --- GUI Elements ---
        self.main_frame = tk.Frame(root, bg="#34495e", bd=5, relief="ridge")
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.title_label = tk.Label(
            self.main_frame,
            text="Emotion Detection Through Voice",
            font=("Arial", 20, "bold"),
            bg="#34495e",
            fg="white",
        )
        self.title_label.pack(pady=10)

        # Result label for displaying emotion
        self.result_label = tk.Label(
            self.main_frame,
            text="Result will appear here",
            font=("Arial", 16),
            bg="#34495e",
            fg="#ecf0f1",
            wraplength=500,
        )
        self.result_label.pack(pady=20, fill="x")

        # Buttons for interaction
        self.button_frame = tk.Frame(self.main_frame, bg="#34495e")
        self.button_frame.pack(pady=10)
        
        self.upload_btn = tk.Button(
            self.button_frame,
            text="Upload Voice Note",
            command=self.upload_audio,
            font=("Arial", 12),
            bg="#2ecc71",
            fg="white",
            padx=10,
            pady=5,
        )
        self.upload_btn.grid(row=0, column=0, padx=10)
        
        self.status_label = tk.Label(
            self.main_frame,
            text="",
            font=("Arial", 10, "italic"),
            bg="#34495e",
            fg="#95a5a6"
        )
        self.status_label.pack(pady=10)

    def load_models(self):
        """Loads the pre-trained gender and emotion models."""
        try:
            self.gender_model = load_model(os.path.join(MODELS_PATH, "gender_model.h5"))
            self.emotion_model = load_model(os.path.join(MODELS_PATH, "emotion_model.h5"))
            
            self.gender_encoder = LabelEncoder()
            self.gender_encoder.fit(GENDERS)
            
            self.emotion_encoder = LabelEncoder()
            self.emotion_encoder.fit(EMOTIONS)
            
            print("Models loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {e}")
            self.root.destroy()

    def extract_features(self, file_name):
        """Extracts audio features from a given file."""
        try:
            X, sample_rate = librosa.load(file_name, sr=SAMPLE_RATE)
            if X.ndim > 1:
                X = X.mean(axis=1)

            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=N_MFCC)
            chroma = librosa.feature.chroma_stft(y=X, sr=sample_rate)
            mel = librosa.feature.melspectrogram(y=X, sr=sample_rate)

            features = np.concatenate((
                np.mean(mfccs.T, axis=0),
                np.mean(chroma.T, axis=0),
                np.mean(mel.T, axis=0)
            ))
            return features
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract features: {e}")
            return None

    def predict_emotion(self, audio_file_path):
        """Processes an audio file, checks gender, and predicts emotion if female."""
        self.status_label.config(text="Processing audio and detecting gender...")
        
        features = self.extract_features(audio_file_path)
        if features is None:
            return

        # Gender Detection
        X_gender = features.reshape(1, features.shape[0], 1)
        gender_pred = self.gender_model.predict(X_gender, verbose=0)
        predicted_gender_idx = np.argmax(gender_pred, axis=1)[0]
        predicted_gender = self.gender_encoder.inverse_transform([predicted_gender_idx])[0]

        if predicted_gender != 'female':
            self.status_label.config(text="Gender not detected as female.")
            messagebox.showinfo("Result", "This model is designed to work exclusively with female voices. Please upload a female voice instead.")
        else:
            self.status_label.config(text="Gender confirmed as female. Detecting emotion...")
            # Emotion Detection
            X_emotion = features.reshape(1, features.shape[0], 1)
            emotion_pred = self.emotion_model.predict(X_emotion, verbose=0)
            predicted_emotion_idx = np.argmax(emotion_pred, axis=1)[0]
            predicted_emotion = self.emotion_encoder.inverse_transform([predicted_emotion_idx])[0]

            self.result_label.config(text=f"Detected Emotion: {predicted_emotion.capitalize()}")
            self.status_label.config(text="Prediction complete.")

    def upload_audio(self):
        """Handles audio file upload and prediction."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac")]
        )
        if file_path:
            self.upload_btn.config(state=tk.DISABLED)
            self.status_label.config(text="File selected. Processing...")
            
            try:
                # I used a thread for file processing to keep the GUI responsive
                processing_thread = threading.Thread(target=lambda: self.predict_emotion(file_path))
                processing_thread.start()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process the uploaded file: {e}")
            finally:
                self.upload_btn.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()
