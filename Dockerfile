FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the MediaPipe face landmarker model
RUN wget -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -O face_landmarker.task

# Copy the rest of the application code
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
