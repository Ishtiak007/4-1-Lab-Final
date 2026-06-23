import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


output_dir = "lab1_outputs"
os.makedirs(output_dir, exist_ok=True)


img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("input.jpg not found. Please keep input.jpg in the same folder.")
    exit()

img = cv2.resize(img, (512, 512))

cv2.imwrite(os.path.join(output_dir, "original_512.png"), img)


# Original Image
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")
plt.savefig(os.path.join(output_dir, "original_display.png"), dpi=150, bbox_inches="tight")
plt.show()


# Spatial Resolution Reduction
resize_values = [512, 256, 128, 64, 32]

plt.figure(figsize=(14, 5))

for index, value in enumerate(resize_values):
    small = cv2.resize(img, (value, value))
    enlarged = cv2.resize(small, (512, 512), interpolation=cv2.INTER_NEAREST)

    plt.subplot(1, 5, index + 1)
    plt.imshow(enlarged, cmap="gray")
    plt.title(f"{value}x{value}")
    plt.axis("off")

plt.suptitle("Spatial Resolution Reduction")
plt.savefig(os.path.join(output_dir, "spatial_resolution.png"), dpi=150, bbox_inches="tight")
plt.show()


# Intensity Level Reduction
bit_depths = [8, 7, 6, 5, 4, 3, 2, 1]

plt.figure(figsize=(14, 7))

for index, bit in enumerate(bit_depths):
    levels = 2 ** bit

    reduced = np.round(img * (levels - 1) / 255)
    reduced = np.uint8(reduced * (255 / (levels - 1)))

    plt.subplot(2, 4, index + 1)
    plt.imshow(reduced, cmap="gray")
    plt.title(f"{bit}-bit\n{levels} levels")
    plt.axis("off")

plt.suptitle("Intensity Level Resolution Reduction")
plt.savefig(os.path.join(output_dir, "intensity_resolution.png"), dpi=150, bbox_inches="tight")
plt.show()


# Histogram
plt.figure(figsize=(8, 5))
plt.hist(img.ravel(), bins=256, range=(0, 256))
plt.title("Histogram of Grayscale Image")
plt.xlabel("Gray Level")
plt.ylabel("Number of Pixels")
plt.savefig(os.path.join(output_dir, "histogram.png"), dpi=150, bbox_inches="tight")
plt.show()


# Threshold Segmentation
threshold = 127
_, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

cv2.imwrite(os.path.join(output_dir, "threshold_segmented.png"), binary_img)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(binary_img, cmap="gray")
plt.title(f"Threshold Segmentation, T = {threshold}")
plt.axis("off")

plt.savefig(os.path.join(output_dir, "threshold_result.png"), dpi=150, bbox_inches="tight")
plt.show()


print("Lab 1 completed successfully.")
print("Check the lab1_outputs folder.")