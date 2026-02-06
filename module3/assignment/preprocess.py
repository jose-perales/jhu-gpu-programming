#!/usr/bin/env python3
"""
Preprocess images for CUDA contrast adjustment assignment.
Loads standard test images from scikit-image, converts to grayscale, saves as binary.

Binary format:
  - width (4 bytes, unsigned int)
  - height (4 bytes, unsigned int)  
  - pixel data (width * height bytes, uint8)
"""

import struct
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import data


# Standard test images from scikit-image
IMAGES = [
    'astronaut', 'brick', 'camera', 'cat', 'checkerboard', 'chelsea', 'clock',
    'coffee', 'coins', 'colorwheel', 'grass', 'gravel', 'hubble_deep_field',
    'immunohistochemistry', 'logo', 'moon', 'page', 'retina', 'rocket',
    'shepp_logan_phantom', 'text',
]


def load_image(name: str) -> np.ndarray | None:
    """Load a standard test image by name."""

    img = getattr(data, name)()
        
    # Convert to grayscale
    if len(img.shape) == 3:
        # RGB/RGBA to grayscale: 0.299*R + 0.587*G + 0.114*B
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    
    # Normalize to 0-255
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    
    return img


def save_binary(img: np.ndarray, output_path: Path):
    """Save image as binary file with header."""

    height, width = img.shape
    
    # Write binary file:
    # [width (4 bytes), height (4 bytes), pixel data]
    with open(output_path, 'wb') as f:
        f.write(struct.pack('II', width, height))
        f.write(img.astype(np.uint8).tobytes())
    
    return width, height


def save_meta(img: np.ndarray, meta_path: Path):
    """Save image metadata to text file."""

    height, width = img.shape
    
    with open(meta_path, 'w') as f:
        f.write(f"{width}\n{height}\n")


def save_image(img: np.ndarray, image_path: Path):
    """Save image as PNG."""

    pil_img = Image.fromarray(img)
    pil_img.save(image_path)

def process_single_image(name: str, output_dir: Path) -> bool:
    """Process a single image and save outputs."""

    img = load_image(name)
    if img is None:
        return False
    
    n_pixels = img.shape[0] * img.shape[1]
    
    # Save outputs
    bin_path = output_dir / f"{name}.bin"
    save_binary(img, bin_path)
    save_meta(img, output_dir / f"{name}.txt")
    save_image(img, output_dir / f"{name}.png")
    
    print(f"  {name}: {img.shape[1]}x{img.shape[0]} = {n_pixels} pixels")
    return True


def main():
    output_dir = Path("images")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Processing {len(IMAGES)} images...\n")
    
    successful = 0
    for name in IMAGES:
        if process_single_image(name, output_dir):
            successful += 1
    
    print(f"\nProcessed {successful} images to {output_dir}/")


if __name__ == "__main__":
    main()
