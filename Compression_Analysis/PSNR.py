import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np


def rgb_to_luminance(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convert RGB channels to luminance using the ITU-R BT.601 standard."""
    return 0.299 * r + 0.587 * g + 0.114 * b


def calculate_luminance(image: np.ndarray) -> np.ndarray:
    """Calculate luminance from a BGR image (OpenCV's default format)."""
    b, g, r = cv2.split(image)
    return rgb_to_luminance(r, g, b)


def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """Calculate PSNR between two luminance arrays (higher values indicate better quality)."""
    mse = np.mean(np.square(original - compressed))
    return float("inf") if mse == 0 else 10 * math.log10((255.0**2) / mse)


def load_image(image_path: str) -> np.ndarray:
    """Load an image with cross-platform path handling and error debugging."""
    # Automatically convert path separators (Windows: \, Linux: /)
    path = Path(image_path).resolve()  # Convert to absolute path for clarity

    # Check if file exists
    if not path.is_file():
        # Show system-specific debug info
        print("\nSystem Debug Info:")
        print(f"Operating System: {'Windows' if os.name == 'nt' else 'Linux/Unix'}")
        print(f"Current Working Directory: {Path.cwd()}")
        print(f"Target Path (auto-converted): {path}")
        raise FileNotFoundError(
            f"Image file not found! Check the path:\n{path}\n"
            f"Note: Path automatically adjusted for your operating system."
        )

    # Load image with OpenCV
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(
            f"Invalid image format: {path} (supported: PNG, JPG, BMP, TIFF)"
        )

    return image


def main() -> None:
    """Cross-platform PSNR calculation tool with automatic path handling."""
    # Define default paths using Path (auto-handles separators)
    default_original: Path = Path("Compression_Analysis") / "orignal_image.png"
    default_compressed: Path = Path("Compression_Analysis") / "compressed_image.png"

    # Set up command-line arguments
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Cross-Platform PSNR Calculator (supports Windows/Linux)"
    )
    parser.add_argument(
        "--original",
        default=str(default_original),
        help=f"Path to original image (default: {default_original})",
    )
    parser.add_argument(
        "--compressed",
        default=str(default_compressed),
        help=f"Path to compressed image (default: {default_compressed})",
    )
    args: argparse.Namespace = parser.parse_args()

    try:
        # Load images with cross-platform path handling
        original_image: np.ndarray = load_image(args.original)
        compressed_image: np.ndarray = load_image(args.compressed)

        # Verify image dimensions match
        if original_image.shape != compressed_image.shape:
            raise ValueError(
                f"Dimension mismatch! Original: {original_image.shape[:2]}, "
                f"Compressed: {compressed_image.shape[:2]}"
            )

        # Calculate and display PSNR
        original_lum: np.ndarray = calculate_luminance(original_image)
        compressed_lum: np.ndarray = calculate_luminance(compressed_image)
        psnr: float = calculate_psnr(original_lum, compressed_lum)

        print(f"PSNR Value: {psnr:.2f} dB")
        print(
            "Interpretation: 20-30 dB = low quality, 30-40 dB = good quality, >40 dB = excellent quality"
        )

    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
