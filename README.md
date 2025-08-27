# Voice Emotion and Gender Detection

This project is a machine learning application that identifies emotion from a human voice. A key feature of this system is its ability to first detect the speaker's gender and then, as required by the assignment, proceed with emotion prediction only if the voice is identified as female.

The application is built to handle both pre-recorded audio files and a live microphone feed, providing a user-friendly way to interact with the models.

## Key Features

* **Emotion Recognition:** The core model classifies a female voice clip into one of seven emotions: **angry**, **calm**, **happy**, **sad**, **fearful**, **surprised**, or **disgusted**.

* **Female Voice Filter:** The application first uses a separate, custom-trained model to determine the speaker's gender. If the voice is not female, it prompts the user with a message.

* **Custom Models:** The solution is built with two distinct neural networks: a **gender detection model** and an **emotion recognition model**.

* **Audio Feature Extraction:** The models are trained on advanced audio features like MFCCs, Chroma, and Mel Spectrograms, which are extracted from the raw audio data using the `librosa` library.

* **Graphical Interface (GUI):** A clean interface built with `tkinter` provides a simple way to upload audio files for analysis.

---

## Project Structure

```
Emotion_Detection_Voice/
├── data/
│   ├── raw/
│   │   └── RAVDESS_Dataset/     (Raw audio files for all training)
│   ├── processed/
│   │   ├── male_features/       (Saved features from male voices)
│   │   └── female_features/     (Saved features from female voices)
├── src/
│   ├── models/
│   │   ├── emotion_model.h5     (Saved emotion detection model)
│   │   └── gender_model.h5      (Saved gender detection model)
│   ├── emotion_training.ipynb   (Jupyter Notebook for all training and evaluation)
│   └── main_app.py              (The main GUI application script)
└── requirements.txt           (All project dependencies)
```

---

## Setup and Installation

### 1. Clone the Repository

If you haven't already, clone the project to your local machine:

### 2. Set Up the Environment

Create and activate a virtual environment to manage dependencies:

```bash
python -m venv venv
venv/Scripts/activate
```

### 3. Install Requirements

Install all the necessary Python libraries:

```bash
pip install -r requirements.txt
```

### 4. Data and Models

* **Download the RAVDESS Dataset** and place the audio files in the `data/raw/RAVDESS_Dataset/` folder.

* Run the `emotion_training.ipynb` notebook to preprocess the data, train both custom models, and save the `.h5` model files to the `src/models/` directory.

## How to Run the Application

Once everything is set up and your models are trained and saved, you can run the main application from your project's root directory:

```bash
python src/main_app.py
```

This will launch the GUI, where you can upload an audio file to detect the speaker's emotion.

### Download Datasets and Models

To get the project fully running, you will need to download the datasets and the custom-trained model weights.

* **Datasets:** You will need to acquire the **RAVDESS** dataset from a reliable source.

* **Custom Models:** The custom-trained model weights (`.h5` files) are too large for GitHub. You can download them from the following link and place them in `src/models/`.

[Google Drive Link for Models: [INSERT_LINK_HERE](https://drive.google.com/drive/folders/17XoyfJeLKhnJzeEXIzAQH9GttNkMXsuV?usp=sharing)]
