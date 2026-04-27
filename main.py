import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CV App Documentation Hub", page_icon="📘", layout="wide")

PAGE_TUTORIALS = {
    "Videos Clipper": """
**Purpose:**
- Serve as the bulk video clipper and batch processing hub.
- Generate clips using a spreadsheet, source folder, and FFMPEG-backed workflow.

**Key Features:**
- Source and output path configuration.
- Spreadsheet upload for CSV / XLSX / XLS.
- Header row selection and dynamic column mapping.
- Column mapping for video filename, start time, end time or duration, main label, and secondary labels.
- Data compilation into a video dictionary with preview and summary metrics.
- Pre-flight validation of video existence, duration, and time bounds.
- Multiple output modes: auto-create folders, grouped output, and subfolder tree replication.
- Error handling, skip-existing logic, and detailed processing logs.

**Workflow:**
1. Upload your spreadsheet and select the source video folder.
2. Map the required columns and compile the data plan.
3. Validate the compiled clips and fix any spreadsheet or folder issues.
4. Start clip processing and review the results table.

**Tip:** Use Mode A to auto-create clip folders next to each source video, or Mode C to preserve original subfolder structure.
""",
    "Transformations": """
**Purpose:**
- Build advanced image/video processing pipelines with step-by-step controls.
- Apply geometric, color, filtering, frequency, and face-based transformations.

**Key Features:**
- Multiple image/video uploads for batch processing.
- Output folder configuration and optional pipeline JSON loading.
- Information category selection for metadata display.
- Add transformation steps from categorized operation groups.
- Reorder, delete, and tune pipeline stages.
- Parameter controls for resizing, rotation, translation, cropping, brightness/contrast, blur, edge detection, noise, and more.
- Face-specific tools like face alignment and advanced face cropping.
- Media filter and video output format selection.

**Workflow:**
1. Upload files and choose the output destination.
2. Optionally load an existing JSON pipeline.
3. Select transformations, add them to the pipeline, and configure each step.
4. Apply the pipeline to the chosen media and export the results.

**Tip:** Save your pipeline to JSON so you can reuse the exact operation sequence later.
""",
    "Labels": """
**Purpose:**
- Annotate images and videos with emotions, intensities, secondary labels, and facial action units.
- Save structured annotation output per file.

**Key Features:**
- Media type filtering and directory path selection.
- Directory initialization for dataset folders.
- Shuffle mode for randomized annotation order.
- Custom emotion creation and persistent emotion list storage.
- Main emotion selection plus secondary emotions.
- Intensity slider with dynamic min/max bounds.
- Detailed Action Unit selectors grouped by face region.
- Optional landmark overlay and hover coordinate preview.
- Annotator notes and JSON export per file.

**Workflow:**
1. Set the target directory and filter media files.
2. Choose the emotion label, intensity, and any additional action units.
3. Navigate through files using previous/next controls.
4. Save annotations to JSON and continue labeling.

**Tip:** Add new emotions using the sidebar expander so they persist in `list_of_emotions.json`.
"""
}

# --- HELPER FUNCTION FOR TUTORIALS ---
def render_tutorial(page_name):
    with st.expander(f"📖 How to use the {page_name} page", expanded=False):
        st.markdown(PAGE_TUTORIALS.get(page_name, "**No tutorial available yet.**"))
        st.markdown(f"**Source file:** `pages/{page_name.lower().replace(' ', '_')}.py`")

# --- PAGE FUNCTIONS ---
def videos_clipper_page():
    st.title("🎬 Videos Clipper")
    st.subheader("Feature Overview")
    st.markdown(
        """
- **Bulk spreadsheet-based clipping** for video datasets.
- **Dynamic column mapping** for start/end times, labels, and metadata.
- **Pre-flight validation** with duration checks and error reporting.
- **Output modes** for folder grouping, automatic clip folders, or replicated subfolder trees.
- **Processing logs** and result summaries after clip creation.
"""
    )
    st.write(
        "This page documents how the video clipper operates, from spreadsheet upload through validation and batch export."
    )
    render_tutorial("Videos Clipper")

def transformations_page():
    st.title("🖼️ Transformations")
    st.subheader("Feature Overview")
    st.markdown(
        """
- **Multi-file upload** for image and video processing.
- **Transform pipeline builder** with categorized operation groups.
- **Interactive step management** for reordering and removing transformations.
- **Per-operation parameter controls** for geometric, color, filtering, noise, and face-aware transforms.
- **Export options** for videos, frames, or both.
"""
    )
    st.write(
        "This page explains the transformation pipeline: selecting operations, configuring parameters, and processing media in batch."
    )
    render_tutorial("Transformations")

def labels_page():
    st.title("🏷️ Labels")
    st.subheader("Feature Overview")
    st.markdown(
        """
- **Emotion annotation** with main and secondary labels.
- **Intensity scaling** with custom min/max bounds.
- **Custom emotion management** and persistence.
- **Action Unit selectors** grouped by face region.
- **Annotator notes** and JSON export for each media file.
"""
    )
    st.write(
        "This page provides a tutorial-style overview of the emotion labeling workflow, including media selection, annotation, and saving."
    )
    render_tutorial("Labels")


# --- MAIN NAVIGATION ---
def main():
    st.sidebar.title("Navigation")
    
    pages = {
        "Videos Clipper": videos_clipper_page,
        "Transformations": transformations_page,
        "Labels": labels_page
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selection]()

if __name__ == "__main__":
    main()