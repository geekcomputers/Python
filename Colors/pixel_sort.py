"""Pixel Sorting with Audio Visualization

This script processes an image by sorting its pixels based on color attributes,
generates a video of the sorting process, and converts pixel data into audio.
"""

import argparse
import colorsys
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

# Configuration - Modify these for different results
AUDIO_SAMPLE_RATE: int = 44100  # Audio sampling rate (Hz)
AUDIO_DURATION: float = 10        # Audio duration (seconds)
BASE_FREQUENCY: float = 220       # Lowest frequency for audio conversion (Hz)
MAX_FREQUENCY: float = 880        # Highest frequency for audio conversion (Hz)
SORT_REPETITIONS: int = 8       # Controls color sorting smoothness
OUTPUT_VIDEO_FPS: int = 16      # Frames per second for output video

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with default values"""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Pixel sorting with audio visualization")
    parser.add_argument("-f", "--filename", required=True, 
                        help="Input image filename (without extension)")
    parser.add_argument("-i", "--input_dir", default="Image", 
                        help="Input directory for images (default: Image)")
    parser.add_argument("-o", "--output_dir", default="Image_sort", 
                        help="Output directory for results (default: Image_sort)")
    parser.add_argument("-d", "--duration", type=float, default=AUDIO_DURATION, 
                        help="Audio duration in seconds (default: 10)")
    return parser.parse_args()

def validate_input_image(args: argparse.Namespace) -> str:
    """Validate input image exists and return its path"""
    image_path: Path = Path(args.input_dir) / f"{args.filename}.jpg"
    if not image_path.exists():
        print(f"Error: Image file '{image_path}' not found")
        exit(1)
    return str(image_path)

def create_output_directories(args: argparse.Namespace) -> Path:
    """Create necessary output directories"""
    output_base: Path = Path(args.output_dir)
    output_subdir: Path = output_base / args.filename
    output_subdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created: {output_subdir}")
    return output_subdir

def sort_pixels_by_row(img: np.ndarray) -> np.ndarray:
    """Sort image pixels row by row using HSV color space"""
    height, width = img.shape[:2]
    sorted_img: np.ndarray = img.copy().astype(np.float32) / 255.0  # Normalize to [0,1]
    
    for row in tqdm(range(height), desc="Sorting rows"):
        # Extract row pixels and sort based on HSV luminance
        row_pixels: np.ndarray = sorted_img[row].copy()
        row_pixels_sorted: np.ndarray = sort_pixels_by_hsv(row_pixels)
        sorted_img[row] = row_pixels_sorted
    
    return (sorted_img * 255).astype(np.uint8)  # Convert back to [0,255]

def sort_pixels_by_hsv(pixels: np.ndarray) -> np.ndarray:
    """Sort pixels using HSV color space for better visual coherence"""
    # Calculate HSV-based sorting key for each pixel
    sort_keys: np.ndarray = np.array([step_sort_key(pixel) for pixel in pixels])
    # Sort pixels based on the computed keys
    sort_indices: np.ndarray = np.argsort(sort_keys, axis=0)[:, 0]
    return pixels[sort_indices]

def step_sort_key(bgr: np.ndarray, repetitions: int = SORT_REPETITIONS) -> tuple[int, float, int]:
    """Generate sort key based on HSV color space and luminance"""
    b, g, r = bgr
    # Calculate luminance (weighted for human perception)
    luminance: float = math.sqrt(0.241 * r + 0.691 * g + 0.068 * b)
    # Convert to HSV for better color sorting
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    # Adjust for repetitions to reduce noise and create smoother bands
    h_scaled: int = int(h * repetitions)
    v_scaled: int = int(v * repetitions)
    # Invert for smoother transitions between colors
    if h_scaled % 2 == 1:
        v_scaled = repetitions - v_scaled
        luminance = repetitions - luminance
    return (h_scaled, luminance, v_scaled)

def generate_sorting_video(original_img: np.ndarray, sorted_img: np.ndarray, output_path: Path, fps: int = OUTPUT_VIDEO_FPS) -> None:
    """Generate video showing the pixel sorting process"""
    print("\n>>> Generating sorting process video...")
    height, width = original_img.shape[:2]
    fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer: cv2.VideoWriter = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Create intermediate frames showing progressive sorting
    for row in tqdm(range(height), desc="Creating video frames"):
        frame: np.ndarray = original_img.copy()
        frame[:row+1] = sorted_img[:row+1]
        video_writer.write(frame)
    
    # Add a few extra frames of the fully sorted image
    for _ in range(fps):
        video_writer.write(sorted_img)
    
    video_writer.release()
    print(f"Video saved to: {output_path}")

def save_pixel_data(pixels: np.ndarray, output_path: Path) -> None:
    """Save pixel data to Excel file for analysis"""
    try:
        df: pd.DataFrame = pd.DataFrame(pixels.reshape(-1, 3), columns=["Blue", "Green", "Red"])
        df.to_excel(str(output_path), index=False)
        print(f"Pixel data saved to: {output_path}")
    except Exception as e:
        print(f"Warning: Could not save pixel data - {e}")

def pixels_to_audio(pixels: np.ndarray, duration: float = AUDIO_DURATION, sample_rate: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
    """Convert pixel data to audio signal"""
    print("\n>>> Generating audio from pixel data...")
    num_samples: int = int(sample_rate * duration)
    
    # Resize pixel data to match audio length
    if pixels.size > 0:
        # Flatten and normalize pixel data
        pixels_flat: np.ndarray = pixels.reshape(-1, 3) / 255.0
        # Interpolate to match audio sample count
        x_old: np.ndarray = np.linspace(0, 1, len(pixels_flat))
        x_new: np.ndarray = np.linspace(0, 1, num_samples)
        pixel_data: np.ndarray = np.array([
            np.interp(x_new, x_old, pixels_flat[:, i]) for i in range(3)
        ]).T
    else:
        # Generate random pixel data if input is empty
        pixel_data: np.ndarray = np.random.rand(num_samples, 3)
    
    # Map pixel channels to audio parameters
    frequencies: np.ndarray = calculate_frequencies(pixel_data)
    amplitudes: np.ndarray = calculate_amplitudes(pixel_data)
    waveforms: np.ndarray = calculate_waveforms(pixel_data)
    
    # Generate audio signal
    t: np.ndarray = np.linspace(0, duration, num_samples, endpoint=False)
    audio_signal: np.ndarray = np.zeros(num_samples)
    
    for i in range(num_samples):
        # Generate waveform based on pixel data
        if waveforms[i] < 0.25:  # Sine wave
            audio_signal[i] = amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t[i])
        elif waveforms[i] < 0.5:  # Square wave
            audio_signal[i] = amplitudes[i] * signal.square(2 * np.pi * frequencies[i] * t[i])
        elif waveforms[i] < 0.75:  # Triangle wave
            audio_signal[i] = amplitudes[i] * signal.sawtooth(2 * np.pi * frequencies[i] * t[i], width=0.5)
        else:  # Sawtooth wave
            audio_signal[i] = amplitudes[i] * signal.sawtooth(2 * np.pi * frequencies[i] * t[i])
    
    # Normalize audio signal
    if np.max(np.abs(audio_signal)) > 0:
        audio_signal /= np.max(np.abs(audio_signal))
    
    return audio_signal

def calculate_frequencies(pixels: np.ndarray) -> np.ndarray:
    """Map pixel red channel to audio frequencies"""
    return BASE_FREQUENCY + pixels[:, 2] * (MAX_FREQUENCY - BASE_FREQUENCY)

def calculate_amplitudes(pixels: np.ndarray) -> np.ndarray:
    """Map pixel green channel to audio amplitudes"""
    return 0.1 + pixels[:, 1] * 0.7  # Range [0.1, 0.8]

def calculate_waveforms(pixels: np.ndarray) -> np.ndarray:
    """Map pixel blue channel to waveform types"""
    return pixels[:, 0]  # Range [0, 1]

def save_audio(audio_signal: np.ndarray, output_path: Path, sample_rate: int = AUDIO_SAMPLE_RATE) -> None:
    """Save audio signal to WAV file"""
    try:
        from scipy.io import wavfile
        # Convert to 16-bit integer format
        audio_int16: np.ndarray = (audio_signal * 32767).astype(np.int16)
        wavfile.write(str(output_path), sample_rate, audio_int16)
        print(f"Audio saved to: {output_path}")
    except Exception as e:
        print(f"Error saving audio: {e}")
        print("Hint: Ensure scipy is installed (pip install scipy)")

def main() -> None:
    """Main processing pipeline"""
    # Parse command line arguments
    args: argparse.Namespace = parse_arguments()
    
    # Validate input and create output directories
    input_path: str = validate_input_image(args)
    output_dir: Path = create_output_directories(args)
    
    # Load and process image
    try:
        img: np.ndarray = cv2.imread(input_path)
        img = cv2.resize(img, (800, 500))
        print(f"Loaded image: {input_path} ({img.shape[1]}x{img.shape[0]})")
    except Exception as e:
        print(f"Error loading image: {e}")
        exit(1)
    
    # Perform pixel sorting
    sorted_img: np.ndarray = sort_pixels_by_row(img)
    
    # Save sorted image
    sorted_img_path: Path = output_dir / f"{args.filename}.jpg"
    cv2.imwrite(str(sorted_img_path), sorted_img)
    print(f"Sorted image saved to: {sorted_img_path}")
    
    # Generate and save video
    video_path: Path = output_dir / f"{args.filename}.mp4"
    generate_sorting_video(img, sorted_img, video_path)
    
    # Save pixel data
    pixel_data_path: Path = output_dir / "pixel_data.xlsx"
    save_pixel_data(sorted_img, pixel_data_path)
    
    # Generate and save audio
    audio_path: Path = output_dir / f"{args.filename}.wav"
    audio_signal: np.ndarray = pixels_to_audio(sorted_img, args.duration)
    save_audio(audio_signal, audio_path)
    
    print("\n=== Processing Complete ===")
    print(f"All results saved to: {output_dir}")

if __name__ == "__main__":
    main()