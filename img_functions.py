import cv2
import numpy as np
from scipy.fftpack import fft2, fftshift
import pywt
import math
from PIL import Image

# --- Geometric ---
def translate_image(img, tx, ty):
    """Shifts the image space using a 2x3 affine matrix."""
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h))

def rotate_image(img, angle):
    """Rotates around the center pixel."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def resize_image(img, width, height, interpolation):
    """Resizes the image to absolute pixel dimensions."""
    interp_flag = getattr(cv2, interpolation)
    # cv2.resize expects dimensions as (width, height)
    return cv2.resize(img, (int(width), int(height)), interpolation=interp_flag)

def shear_image(img, shx, shy):
    """Applies a shear transformation via off-diagonal matrix manipulation."""
    h, w = img.shape[:2]
    M = np.float32([[1, shx, 0], [shy, 1, 0]])
    return cv2.warpAffine(img, M, (w, h))

def flip_image(img, mode):
    """Flips image axes. Horizontal, Vertical, or Both."""
    mode_map = {"Horizontal": 1, "Vertical": 0, "Both": -1}
    flip_code = mode_map.get(mode, 1)
    return cv2.flip(img, flip_code)

def crop_image(img, x, y, w, h):
    """Slices the numpy array safely inside image bounds."""
    img_h, img_w = img.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return img[y:y+h, x:x+w]

# --- Photometric ---
def adjust_brightness_contrast(img, alpha, beta):
    """
    Scales intensities and adds a bias: $g(x,y) = \alpha f(x,y) + \beta$
    Explicitly clips values to prevent looping/overflow artifacts.
    """
    # Convert to float32 first. If we stay in uint8, 250 + 10 = 4 (overflow)
    adjusted = img.astype(np.float32) * alpha + beta
    # Clip values below 0 to 0, and above 255 to 255
    clipped = np.clip(adjusted, 0, 255)
    return clipped.astype(np.uint8)

def apply_gamma(img, gamma):
    """Non-linear intensity transformation using a Lookup Table (LUT) for speed."""
    invGamma = 1.0 / gamma if gamma != 0 else 1.0
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_histogram_equalization(img, method):
    """Spreads out the most frequent intensity values."""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        if method == 'Global':
            l = cv2.equalizeHist(l)
        else: # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
        merged = cv2.merge((l, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    else:
        return cv2.equalizeHist(img)

def apply_thresholding(img, thresh=127, type="Binary"):
    """Binarizes the image using a fixed threshold and selected threshold type."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    type_map = {
        "Binary": cv2.THRESH_BINARY,
        "Binary Inverted": cv2.THRESH_BINARY_INV,
        "Truncate": cv2.THRESH_TRUNC,
        "To Zero": cv2.THRESH_TOZERO,
        "To Zero Inverted": cv2.THRESH_TOZERO_INV,
    }
    flag = type_map.get(type, cv2.THRESH_BINARY)
    _, thresh_img = cv2.threshold(gray, thresh, 255, flag)
    return cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2RGB)

def apply_posterization(img, bits):
    """Reduces the color palette by right-shifting bits to drop precision."""
    shift = 8 - bits
    return np.uint8((img >> shift) << shift)

def invert_image(img):
    """Subtracts pixel values from max range (255)."""
    return cv2.bitwise_not(img)

def square_image(img, mode="Crop"):
    """
    Forces the image into a perfect 1:1 aspect ratio without distorting/squishing the contents.
    """
    h, w = img.shape[:2]
    
    # If the image is already a perfect square, return it immediately to save compute time
    if h == w:
        return img
        
    if mode == "Crop":
        # Find the shortest dimension to determine the square's size
        side = min(h, w)
        
        # Calculate the exact center of the original image
        cx, cy = w // 2, h // 2
        
        # Calculate the top-left corner of the crop box
        x1 = cx - side // 2
        y1 = cy - side // 2
        
        # Slice the array. (e.g., if image is 1920x1080, side is 1080. Crop keeps the middle 1080 pixels of the width).
        return img[y1:y1+side, x1:x1+side]
        
    elif mode == "Pad":
        # Find the longest dimension to determine the canvas size
        side = max(h, w)
        
        # Create a blank black canvas of the new square size, matching the original image's color channels and data type
        if len(img.shape) == 3:
            canvas = np.zeros((side, side, img.shape[2]), dtype=img.dtype)
        else:
            canvas = np.zeros((side, side), dtype=img.dtype)
            
        # Calculate where to place the top-left corner of the original image so it sits exactly in the center
        x_offset = (side - w) // 2
        y_offset = (side - h) // 2
        
        # Paste the original image array into the blank canvas array
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img
        return canvas
        
    return img

# --- Filtering/Morphology ---
def apply_gaussian_blur(img, ksize):
    """Convolves a Gaussian kernel to reduce high-frequency noise."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def apply_median_blur(img, ksize):
    """Replaces pixels with the median of their neighborhood (non-linear)."""
    return cv2.medianBlur(img, ksize)

def apply_unsharp_mask(img):
    """Subtracts a blurred version from the original to enhance edge contrast."""
    blurred = cv2.GaussianBlur(img, (9, 9), 10.0)
    return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

def apply_canny(img, t1, t2):
    """Multi-stage algorithm to detect optimal continuous edges."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, t1, t2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def apply_morphology(img, operation, shape, ksize):
    """Probes the image with a structuring element to expand/shrink boundaries."""
    shape_map = {"Rect": cv2.MORPH_RECT, "Ellipse": cv2.MORPH_ELLIPSE, "Cross": cv2.MORPH_CROSS}
    op_map = {"Erosion": cv2.MORPH_ERODE, "Dilation": cv2.MORPH_DILATE,
              "Opening": cv2.MORPH_OPEN, "Closing": cv2.MORPH_CLOSE,
              "Gradient": cv2.MORPH_GRADIENT}
    kernel_size = (ksize, ksize) if isinstance(ksize, int) else ksize
    kernel = cv2.getStructuringElement(shape_map[shape], kernel_size)
    return cv2.morphologyEx(img, op_map[operation], kernel)

# --- Frequency Domain ---
def apply_fft_magnitude(img):
    """Maps spatial coordinates to frequency components."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    f = fft2(gray)
    fshift = fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum = np.uint8(cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX))
    return cv2.cvtColor(magnitude_spectrum, cv2.COLOR_GRAY2RGB)

def apply_fourier_transform(img):
    """Alias for Fourier magnitude visualization."""
    return apply_fft_magnitude(img)

def apply_dwt(img, wavelet):
    """Extracts scale-specific frequencies (approximations and details)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    coeffs2 = pywt.dwt2(gray, wavelet)
    LL, _ = coeffs2
    LL_norm = np.uint8(cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX))
    return cv2.cvtColor(LL_norm, cv2.COLOR_GRAY2RGB)

# --- Noise ---
def add_gaussian_noise(img, mean=0.0, std=0.1):
    """Simulates statistical sensor noise."""
    if len(img.shape) == 3:
        row, col, ch = img.shape
        noise_shape = (row, col, ch)
    else:
        row, col = img.shape
        noise_shape = (row, col)
    gauss = np.random.normal(mean, std, noise_shape).astype(np.float32)
    noisy = cv2.add(img.astype(np.float32), gauss)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper(img, amount):
    """Simulates impulse noise / dead sensor pixels."""
    noisy = np.copy(img)
    num_salt = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy[tuple(coords)] = 255
    num_pepper = np.ceil(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy[tuple(coords)] = 0
    return noisy

def add_salt_pepper_noise(img, salt_prob, pepper_prob):
    """Simulates impulse noise using separate salt and pepper probabilities."""
    noisy = np.copy(img)
    img_h, img_w = img.shape[:2]
    num_salt = int(np.ceil(salt_prob * img_h * img_w))
    num_pepper = int(np.ceil(pepper_prob * img_h * img_w))

    salt_coords = [np.random.randint(0, img_h, num_salt), np.random.randint(0, img_w, num_salt)]
    if img.ndim == 2:
        noisy[salt_coords[0], salt_coords[1]] = 255
    else:
        noisy[salt_coords[0], salt_coords[1], :] = 255

    pepper_coords = [np.random.randint(0, img_h, num_pepper), np.random.randint(0, img_w, num_pepper)]
    if img.ndim == 2:
        noisy[pepper_coords[0], pepper_coords[1]] = 0
    else:
        noisy[pepper_coords[0], pepper_coords[1], :] = 0

    return noisy

def add_jpeg_artifacts(img, quality):
    """Forces low-bitrate compression artifacts via memory encoding."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    source = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if len(img.shape) == 3 else img
    _, encimg = cv2.imencode('.jpg', source, encode_param)
    decoded = cv2.imdecode(encimg, cv2.IMREAD_COLOR if len(img.shape) == 3 else cv2.IMREAD_GRAYSCALE)
    if len(img.shape) == 3:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return decoded


# --- Image Statistics Extractors ---
def get_image_stats(img_array, pil_image=None, file_bytes=None):
    """Calculates requested details from the image array and original PIL object."""
    stats = {}
    h, w = img_array.shape[:2]
    is_color = len(img_array.shape) == 3
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if is_color else img_array

    # 1. Dimensional
    gcd = math.gcd(w, h)
    stats['Resolution'] = f"{w} x {h}"
    stats['Aspect Ratio'] = f"{w//gcd}:{h//gcd}"
    stats['Orientation'] = "Square" if w == h else ("Landscape" if w > h else "Portrait")
    
    # 2. Color
    stats['Channels'] = 3 if is_color else 1
    stats['Color Model'] = "RGB" if is_color else "Grayscale"
    
    # 3. Photographic
    stats['Luminance (Mean)'] = f"{np.mean(gray):.2f}"
    stats['Contrast (Std Dev)'] = f"{np.std(gray):.2f}"
    stats['Sharpness (Variance of Laplacian)'] = f"{np.var(cv2.Laplacian(gray, cv2.CV_64F)):.2f}"
    
    # 4. Structural
    counts, _ = np.histogram(gray, bins=256, range=(0, 256))
    probs = counts / sum(counts)
    probs = probs[probs > 0]
    stats['Entropy'] = f"{-sum(probs * np.log2(probs)):.2f}"
    stats['Histogram Data'] = counts # For rendering charts later
    
    # 5. Encapsulation & Metadata (Only available for original PIL image)
    if pil_image:
        stats['File Format'] = pil_image.format
        stats['Color Space'] = pil_image.mode
        stats['Weight'] = f"{len(file_bytes) / 1024:.2f} KB" if file_bytes else "Unknown"
        exif_data = pil_image.getexif()
        stats['EXIF'] = "Present" if exif_data else "None"
    else:
        stats['File Format'] = "N/A (In-Memory Array)"
        stats['Color Space'] = "N/A"
        # Estimate weight by encoding to JPEG in memory
        _, encimg = cv2.imencode('.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) if is_color else img_array)
        stats['Weight'] = f"~{len(encimg.tobytes()) / 1024:.2f} KB (Estimated JPG)"
        stats['EXIF'] = "Stripped by processing"

    return stats


# --- Face Landmarking ---

import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import streamlit as st
import plotly.graph_objects as go

@st.cache_resource
def get_face_landmarker():
    """
    Downloads the required MediaPipe model if missing, 
    and initializes the modern Tasks Vision API.
    Cached by Streamlit to prevent reloading the model on every UI click.
    """
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        with st.spinner("Downloading MediaPipe Face Landmarker model..."):
            urllib.request.urlretrieve(url, model_path)
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

def get_face_landmarks(img):
    """
    Runs MediaPipe ONCE on the original image. 
    Returns the raw global landmarks if a face is found.
    """
    landmarker = get_face_landmarker()
    # Ensure contiguous array for MediaPipe safety
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img))
    results = landmarker.detect(mp_image)
    
    if not results.face_landmarks:
        return None
        
    h, w = img.shape[:2]
    return [[int(pt.x * w), int(pt.y * h)] for pt in results.face_landmarks[0]]

def crop_and_map_landmarks(img, global_landmarks, bb_type, **kwargs):
    """
    Calculates the bounding box, crops the image, and mathematically 
    translates the landmarks into the new cropped coordinate space.
    """
    h, w = img.shape[:2]
    
    # 1. Get raw bounds from the global landmarks
    x_coords = [pt[0] for pt in global_landmarks]
    y_coords = [pt[1] for pt in global_landmarks]
    min_x, max_x, min_y, max_y = min(x_coords), max(x_coords), min(y_coords), max(y_coords)
    
    fw, fh = max_x - min_x, max_y - min_y
    cx, cy = min_x + fw // 2, min_y + fh // 2 
    
    # 2. Calculate Bounding Box
    if bb_type == "Minimum":
        x1, y1, x2, y2 = min_x, min_y, max_x, max_y
    elif bb_type == "Square":
        side = max(fw, fh)
        x1, y1 = cx - side // 2, cy - side // 2
        x2, y2 = cx + side // 2, cy + side // 2
    elif bb_type == "Custom":
        cw, ch = kwargs.get('custom_w', fw), kwargs.get('custom_h', fh)
        x1, y1 = cx - cw // 2, cy - ch // 2
        x2, y2 = cx + cw // 2, cy + ch // 2
    elif bb_type == "Oval (Oversize)":
        padding = kwargs.get('oversize_pct', 20) / 100.0
        x1 = int(min_x - (fw * padding))
        y1 = int(min_y - (fh * padding))
        x2 = int(max_x + (fw * padding))
        y2 = int(max_y + (fh * padding))
    
    # Boundary safety checks
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # 3. Apply Crop
    cropped = img[y1:y2, x1:x2]
    
    # 4. Mathematically map landmarks to the new cropped image
    mapped_landmarks = [[lx - x1, ly - y1] for lx, ly in global_landmarks]
    
    return cropped, mapped_landmarks

def extract_landmarks(cropped_img):
    """
    3. Extracts the 478 facial landmarks (updated from 468 to include irises).
    """
    landmarker = get_face_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_img)
    results = landmarker.detect(mp_image)
    
    if not results.face_landmarks:
        return None
        
    h, w = cropped_img.shape[:2]
    landmarks = [[int(pt.x * w), int(pt.y * h)] for pt in results.face_landmarks[0]]
    return landmarks

def advanced_crop_face(img, bb_type="Minimum Rectangle", padding=0, exterior_mode="Set Exterior to 0", poly_string=""):
    """
    Advanced face isolation using binary masking, morphological padding, and exterior handling.
    """
    landmarks = get_face_landmarks(img)
    if not landmarks:
        return img # Graceful fallback if no face is found

    h, w = img.shape[:2]
    
    # 1. Initialize a pure black binary mask matching the image dimensions
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Extract all X and Y coordinates for easy min/max calculations
    x_coords = [pt[0] for pt in landmarks]
    y_coords = [pt[1] for pt in landmarks]
    min_x, max_x = max(0, min(x_coords)), min(w, max(x_coords))
    min_y, max_y = max(0, min(y_coords)), min(h, max(y_coords))
    
    fw, fh = max_x - min_x, max_y - min_y
    cx, cy = min_x + fw // 2, min_y + fh // 2

    # 2. Draw the requested shape onto the mask in pure white (255)
    if bb_type == "Minimum Rectangle":
        cv2.rectangle(mask, (min_x, min_y), (max_x, max_y), 255, -1)
        
    elif bb_type == "Minimum Square":
        side = max(fw, fh)
        sq_x1, sq_y1 = max(0, cx - side // 2), max(0, cy - side // 2)
        sq_x2, sq_y2 = min(w, cx + side // 2), min(h, cy + side // 2)
        cv2.rectangle(mask, (sq_x1, sq_y1), (sq_x2, sq_y2), 255, -1)
        
    elif bb_type == "Minimum Oval":
        # axes expects (width_radius, height_radius)
        axes = (fw // 2, fh // 2)
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
        
    elif bb_type == "Polygonal":
        try:
            # Parse the comma-separated string of IDs into integers
            ids = [int(i.strip()) for i in poly_string.split(",") if i.strip().isdigit()]
            # Ensure valid IDs within the 478 MediaPipe points, and at least 3 points for a polygon
            valid_pts = [landmarks[i] for i in ids if 0 <= i <= 477]
            if len(valid_pts) >= 3:
                # Fill the polygon defined by the user's points
                pts_array = np.array(valid_pts, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts_array], 255)
            else:
                # Failsafe: Fall back to rectangle if input is invalid
                cv2.rectangle(mask, (min_x, min_y), (max_x, max_y), 255, -1)
        except Exception:
            cv2.rectangle(mask, (min_x, min_y), (max_x, max_y), 255, -1)

    # 3. Add Padding using Morphological Dilation
    if padding > 0:
        # Create a structural element. A circular/elliptical kernel creates smoother padded corners.
        kernel_size = padding * 2 + 1 # Must be odd
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # Dilate pushes the white pixels of the mask outwards by the padding amount
        mask = cv2.dilate(mask, kernel, iterations=1)

    # 4. Handle the Exterior
    # Bitwise AND keeps the image pixels where the mask is > 0, setting the rest to pure black.
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    if exterior_mode == "Set Exterior to 0":
        # Return the blacked-out image maintaining original dimensions
        return masked_img
        
    elif exterior_mode == "Cut Out":
        # Find the bounding box of the final padded mask to crop the array physically
        x, y, w_box, h_box = cv2.boundingRect(mask)
        # Slicing the array removes the empty space completely
        return masked_img[y:y+h_box, x:x+w_box]

    return masked_img

def create_interactive_mesh(img, landmarks):
    """
    4. Uses Plotly to overlay landmarks on the image. 
    Allows zooming, panning, and hovering to see the exact Landmark ID.
    """
    h, w = img.shape[:2]
    # Extract X and Y for Plotly
    x = [pt[0] for pt in landmarks]
    y = [h - pt[1] for pt in landmarks] # Plotly Y-axis is inverted compared to OpenCV
    text = [f"ID: {i}" for i in range(len(landmarks))] # Hover text
    
    fig = go.Figure()
    # Add image as background
    fig.add_layout_image(
        dict(source=Image.fromarray(img), x=0, y=h, xref="x", yref="y",
             sizex=w, sizey=h, sizing="stretch", opacity=1, layer="below")
    )
    # Add landmarks as scatter points
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', text=text,
                             marker=dict(size=4, color='red'), hoverinfo='text'))
                             
    fig.update_layout(xaxis=dict(range=[0, w], visible=False),
                      yaxis=dict(range=[0, h], visible=False, scaleanchor="x"),
                      margin=dict(l=0, r=0, t=0, b=0), width=500, height=500)
    return fig

def align_face_by_two_points(img, landmarks, id1, id2, target1, target2):
    """
    5. Computes an affine similarity transform (Rotation, Translation, Scale).
    cv2.estimateAffinePartial2D requires >= 2 points to calculate this mapping.
    """
    src_pts = np.float32([landmarks[id1], landmarks[id2]])
    dst_pts = np.float32([target1, target2])
    
    # Calculate the optimal transformation matrix to move src_pts to dst_pts
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    
    if M is None: return img # Failsafe if math cannot converge
    
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))

def align_face(img, id1, id2, t1_x, t1_y, t2_x, t2_y):
    """Pipeline wrapper for face alignment."""
    landmarks = get_face_landmarks(img)
    if not landmarks: 
        return img
    return align_face_by_two_points(img, landmarks, id1, id2, (t1_x, t1_y), (t2_x, t2_y))

def create_main_preview(img, landmarks=None, highlight_ids=None):
    """Renders the image in Plotly and conditionally overlays the face mesh."""
    h, w = img.shape[:2]
    fig = go.Figure()
    
    # Add image to the background
    fig.add_layout_image(
        dict(source=Image.fromarray(img), x=0, y=h, xref="x", yref="y",
             sizex=w, sizey=h, sizing="stretch", opacity=1, layer="below")
    )
    
    # Conditionally add the mesh
    if landmarks:
        x = [pt[0] for pt in landmarks]
        y = [h - pt[1] for pt in landmarks] # Invert Y for Plotly Cartesian plane
        text = [f"ID: {i}" for i in range(len(landmarks))]
        
        # Base styling for all landmarks
        colors = ['red'] * len(landmarks)
        sizes = [4] * len(landmarks)
        
        # Highlight specifically selected landmarks
        if highlight_ids:
            for hid in highlight_ids:
                if 0 <= hid < len(colors):
                    colors[hid] = 'lawngreen'
                    sizes[hid] = 10
                    text[hid] = f"ID: {hid} (SELECTED)"

        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', text=text,
                                 marker=dict(size=sizes, color=colors), hoverinfo='text'))
                                 
    fig.update_layout(xaxis=dict(range=[0, w], visible=False),
                      yaxis=dict(range=[0, h], visible=False, scaleanchor="x"),
                      margin=dict(l=0, r=0, t=0, b=0), height=600)
    return fig
