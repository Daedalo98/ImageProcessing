# 📘 CV App: Computer Vision Operations Hub

Welcome to **CV App**, a comprehensive Streamlit-based computer vision application designed to simplify and accelerate your media-processing workflows. Whether you need to bulk-clip videos, experiment with complex image transformation pipelines, or carefully label datasets for machine learning, this hub provides a user-friendly visual interface.

---

## ✨ Core Modules

This application is divided into three primary modules, accessible via the sidebar navigation:

### 🎬 Videos Clipper
A bulk-processing engine to slice video datasets accurately using a spreadsheet.
- **Spreadsheet Driven:** Upload an Excel (`.xlsx`, `.xls`) or `.csv` file containing the clip metadata.
- **Dynamic Field Mapping:** Easily map your spreadsheet columns to required parameters (filename, start/end times, labels).
- **Intelligent Pre-flight:** The app validates video existence, checks for duration errors, and validates time bounds before making a single cut.
- **Flexible Exports:** Choose to group clips by source file, auto-create subfolders, or perfectly replicate your original directory tree.

### 🖼️ Transformations
A powerful, step-by-step pipeline builder to apply various visual operations to images and videos.
- **Batch Capabilities:** Upload multiple files at once.
- **Pipeline Builder:** Stack transformations logically. You can apply geometric shifts, adjust colors, apply filters (like Gaussian Blur or Canny Edge Detection), simulate noise, and more.
- **Face-Aware Tools:** Perform advanced operations like face-cropping and facial alignment using MediaPipe landmarks.
- **Save & Load:** Export your exact transformation pipeline as a JSON file to easily reload and re-apply it to future data.

### 🏷️ Labels
A dedicated annotation environment to tag images and videos for emotion and landmark datasets.
- **Emotion Tagging:** Assign main emotions, secondary emotions, and intensity levels. 
- **Action Units:** Log highly specific facial action units (e.g., inner brow raiser, lip corner puller) categorized by facial region.
- **Custom Persistence:** Add new custom emotions directly in the app, which are permanently saved to `list_of_emotions.json`.
- **Output:** Saves a clean, structured JSON file next to your media files containing all entered annotations.

---

## 🚀 How to Run the App (Non-Tech Tutorial)

The easiest and safest way to use this app is through **Docker**. Docker acts like a virtual box that contains everything the app needs to run perfectly (libraries, dependencies, background engines) so you don't have to install complicated programming tools on your own computer.

### Method 1: Using Docker (Recommended)

**Step 1: Install Docker Desktop**
- Go to the [Docker Desktop download page](https://www.docker.com/products/docker-desktop/).
- Download and install the version for your computer (Windows, Mac, or Linux).
- Once installed, open the Docker Desktop application and wait for it to show that the "Engine is running".

**Step 2: Get the Project Folder**
- Download or clone this project folder to your computer.
- Open your computer's Terminal (Mac/Linux) or Command Prompt / PowerShell (Windows).
- Navigate into the downloaded folder. For example:
  ```bash
  cd path/to/your/cv_app
  ```

**Step 3: Start the App**
- In your terminal window, type the following command exactly and press Enter:
  ```bash
  docker compose up --build
  ```
- *Note: The very first time you do this, it may take 5-10 minutes to download all the safe virtual packages and the AI models. Subsequent launches will take only seconds.*

**Step 4: Open the App in your Browser**
- Once the terminal stops printing new lines and you see a message saying "Network cv_app_default  Created", open your favorite web browser (Chrome, Firefox, Safari).
- Go to this address: **[http://localhost:8501](http://localhost:8501)**
- You are now inside the app!

**Step 5: How to Stop the App**
- When you are done using the app, go back to your Terminal window where it is running.
- Press `Ctrl + C` on your keyboard.
- It will safely shut down the virtual box.

> **💡 Note on Files:** Anything you generate in the app (like chopped videos or formatted images) will safely save to the `output_images` folder right inside your project directory on your normal computer.

---

### Method 2: Manual Python Installation (For Developers)

If you are a developer and wish to run the code directly on your host machine:

1. Create a virtual environment to protect your global Python packages:
   ```bash
   python3 -m venv .venv
   source ./.venv/bin/activate  # On Windows use: .\.venv\Scripts\activate
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The provided `requirements.txt` uses `opencv-python-headless` designed for server/docker use. If you need UI elements from OpenCV on your local machine, you may swap it for `opencv-python`.)*
3. Launch the Streamlit server:
   ```bash
   streamlit run main.py
   ```

---

## 📂 Project Structure

- `main.py`: Entry point and documentation hub for the app.
- `pages/`: Contains the individual module screens (`videos_clipper.py`, `transformations.py`, `labels.py`).
- `img_functions.py`: Holds the core mathematical, OpenCV, and MediaPipe logic for image operations.
- `video_functions.py`: Holds logic for decoding spreadsheets and manipulating FFMPEG streams.
- `Dockerfile` & `docker-compose.yml`: Instructions for building the virtual container environment.
- `list_of_emotions.json`: A dynamic list of tags used for the Labels module.
- `output_images/`: The default directory linked to your local machine for retrieving processed files.

---

## 🛠️ Troubleshooting

- **App won't load on localhost:8501:** Ensure Docker Desktop is actually running in the background before you type the start command.
- **"Port is already allocated" error:** Another app is using port 8501. Stop that app first, or change the port mapping in `docker-compose.yml`.
- **Can't find processed files:** When using Docker, ensure you selected `output_images` (or subfolders inside it) as your target destination in the app, as this is the folder bridged back to your actual computer.
