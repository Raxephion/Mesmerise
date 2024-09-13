import numpy as np
from PIL import Image, ImageOps
import cv2
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Function to generate a smooth wavy pattern
def generate_wavy_pattern(X, Y, phase_shift_x, phase_shift_y, amplitude, frequency_x, frequency_y):
    Z = amplitude * np.sin(frequency_x * X + phase_shift_x) * np.cos(frequency_y * Y + phase_shift_y)
    return Z

# Create a grid of coordinates
def create_grid(grid_size):
    x = np.linspace(-2 * np.pi, 2 * np.pi, grid_size)
    y = np.linspace(-2 * np.pi, 2 * np.pi, grid_size)
    X, Y = np.meshgrid(x, y)
    return X, Y

# Generate and normalize the pattern with random parameters
def generate_pattern(grid_size):
    X, Y = create_grid(grid_size)
    phase_shift_x = random.uniform(0, 2 * np.pi)
    phase_shift_y = random.uniform(0, 2 * np.pi)
    amplitude = random.uniform(1, 5)
    frequency_x = random.uniform(0.5, 2.0)
    frequency_y = random.uniform(0.5, 2.0)
    Z = generate_wavy_pattern(X, Y, phase_shift_x, phase_shift_y, amplitude, frequency_x, frequency_y)
    Z_min, Z_max = Z.min(), Z.max()
    Z_normalized = (Z - Z_min) / (Z_max - Z_min)
    return Z_normalized

# Create a gradient colormap with harmonious colors
def create_gradient_colormap():
    colors = [
        "#00008B",  # Deep Blue
        "#00FFFF",  # Cyan
        "#008080",  # Teal
        "#008000",  # Green
        "#00FF00",  # Lime
        "#FFFF00",  # Yellow
        "#FFA500",  # Orange
        "#FF0000"   # Red
    ]
    random.shuffle(colors)  # Shuffle colors to increase randomness
    cmap = LinearSegmentedColormap.from_list("harmonious_cmap", colors)
    return cmap

# Apply the gradient colormap to the pattern
def apply_colormap(Z_normalized, cmap):
    norm = plt.Normalize(Z_normalized.min(), Z_normalized.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colored_image = sm.to_rgba(Z_normalized, bytes=True)[:, :, :3]  # Remove alpha channel
    return colored_image

# Create the kaleidoscope pattern
def create_kaleidoscope_pattern(colored_image):
    image_pil = Image.fromarray(colored_image)
    height, width = colored_image.shape[:2]
    tile_size = height // 4

    kaleidoscope_image = Image.new("RGB", (width, height))

    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            quarter_image = image_pil.crop((j, i, j + tile_size, i + tile_size)).resize((tile_size, tile_size))
            kaleidoscope_image.paste(quarter_image, (j, i))
            kaleidoscope_image.paste(ImageOps.mirror(quarter_image), (width - j - tile_size, i))
            kaleidoscope_image.paste(ImageOps.flip(quarter_image), (j, height - i - tile_size))
            kaleidoscope_image.paste(ImageOps.mirror(ImageOps.flip(quarter_image)), (width - j - tile_size, height - i - tile_size))
    
    return kaleidoscope_image

# Parameters
grid_size = 800

# Generate pattern
Z_normalized = generate_pattern(grid_size)

# Create gradient colormap
cmap = create_gradient_colormap()

# Apply colormap to the pattern
colored_image = apply_colormap(Z_normalized, cmap)

# Create kaleidoscope image
kaleidoscope_image = create_kaleidoscope_pattern(colored_image)

# Save the image using Pillow
kaleidoscope_image.save("KaleidoscopeArtPieceHarmonious.png")

# Optionally, display the image using OpenCV
image_cv = cv2.cvtColor(np.array(kaleidoscope_image), cv2.COLOR_RGB2BGR)
cv2.imshow("Kaleidoscope Harmonious Art Piece", image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()