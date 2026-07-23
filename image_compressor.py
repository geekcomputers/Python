import os
import sys
from PIL import Image

def compress_image(image_path, quality=60):
    """
    Compresses an image by reducing its quality.
    
    Args:
        image_path (str): Path to the image file.
        quality (int): Quality of the output image (1-100). Default is 60.
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Check if file is an image
            if img.format not in ["JPEG", "PNG", "JPG"]:
                print(f"Skipping {image_path}: Not a standard image format.")
                return

            # Create output filename
            filename, ext = os.path.splitext(image_path)
            output_path = f"{filename}_compressed{ext}"

            # Save with reduced quality
            # Optimize=True ensures the encoder does extra work to minimize size
            img.save(output_path, quality=quality, optimize=True)
            
            # Calculate savings
            original_size = os.path.getsize(image_path)
            new_size = os.path.getsize(output_path)
            savings = ((original_size - new_size) / original_size) * 100
            
            print(f"[+] Compressed: {output_path}")
            print(f"    Original: {original_size/1024:.2f} KB")
            print(f"    New:      {new_size/1024:.2f} KB")
            print(f"    Saved:    {savings:.2f}%")

    except Exception as e:
        print(f"[-] Error compressing {image_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python image_compressor.py <image_file>")
        print("Example: python image_compressor.py photo.jpg")
    else:
        target_file = sys.argv[1]
        if os.path.exists(target_file):
            compress_image(target_file)
        else:
            print(f"Error: File '{target_file}' not found.")