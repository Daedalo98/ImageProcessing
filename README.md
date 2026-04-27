# CV App

A Streamlit-based computer vision application for video clipping, image/video transformation pipelines, and emotion/landmark labeling.

## Overview

This project provides three main application modules:

- **Videos Clipper**: Batch-clip videos using a spreadsheet-based workflow with start/end times, labels, and output mode configuration.
- **Transformations**: Build and run reusable image/video transformation pipelines with geometric, color, filtering, noise, and face-aware operations.
- **Labels**: Annotate images and videos with emotion labels, intensity, secondary emotions, facial action units, and save structured JSON output.

The app is designed to support dataset preparation, augmentation, annotation, and clip extraction for computer vision experiments.

## Main Features

### Videos Clipper

- Upload a spreadsheet (`CSV`, `XLSX`, `XLS`) with clip metadata.
- Select the source video folder and choose output storage mode.
- Map spreadsheet columns to video filename, start time, end time or duration, main label, and secondary labels.
- Compile a clip schedule and preview dictionary structure.
- Pre-flight validation of video files, durations, and timing boundaries.
- Automatically create output folders, group clips by source file, or replicate original folder trees.
- Process clips using FFMPEG and review detailed progress logs.

### Transformations

- Upload multiple images or videos for batch transformation.
- Configure an output folder and optionally load a saved pipeline JSON.
- Select information display categories such as dimensional, color, photographic, structural, and metadata.
- Build transformation pipelines from categories including:
  - Geometric transforms: translate, rotate, scale, resize, shear, square, flip, crop.
  - Color/photo transforms: brightness/contrast, gamma, histogram equalization, thresholding, posterization, inversion.
  - Filtering transforms: Gaussian blur, median blur, unsharp mask, Canny edge detection, morphology.
  - Frequency/noise transforms: DWT, Gaussian noise, salt-and-pepper noise, JPEG artifact simulation.
  - Face transforms: advanced face cropping and face alignment.
- Manage pipeline order, remove steps, and configure per-step parameters.
- Export processed media with flexible video/frame output options.

### Labels

- Filter media by images, videos, or both.
- Initialize and validate dataset directories.
- Enable random file order for annotation workflows.
- Choose a main emotion, secondary emotions, and intensity range.
- Add custom emotions that persist across sessions via `list_of_emotions.json`.
- Select facial action units grouped by upper face, eyes/eyelids, lower face, miscellaneous, and head orientation.
- Display optional landmark overlays and coordinate hover details.
- Add annotator notes and save annotations as JSON per file.

## Requirements

Create virtual environment to protect conflicts with your PC's main environment:
```bash
python3 -m venv .venv
source ./.venv/bin/activate
```

Install the required Python packages using the included `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Running the App

From the project root, launch the Streamlit app:

```bash
streamlit run main.py
```

Then open the URL printed in the terminal to access the interface.

## Project Structure

- `main.py` — entry point and documentation hub for the app.
- `pages/videos_clipper.py` — video clipping and spreadsheet-driven batch processing.
- `pages/transformations.py` — transformation pipeline builder and processor.
- `pages/labels.py` — emotion and landmark annotation interface.
- `img_functions.py` — helper functions for image processing and landmark rendering.
- `video_functions.py` — helper functions for loading spreadsheets and processing video clips.
- `list_of_emotions.json` — persisted emotion list used by the labeling module.
- `requirements.txt` — package dependencies.

## Notes

- The app uses Streamlit for the UI and relies on OpenCV, MediaPipe, and PIL for media processing.
- Video clipping uses `imageio-ffmpeg` to invoke FFMPEG for reliable segment extraction.
- Excel spreadsheets are supported through `openpyxl` and `pandas`.

## Troubleshooting

- If Streamlit fails to start, verify the virtual environment is active and `requirements.txt` is installed.
- For missing media files, confirm the source folder path and allowed extensions are correct.
- If spreadsheet columns fail to map, ensure header row selection matches the actual sheet layout.
