import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


output_dir = "lab3_outputs"
os.makedirs(output_dir, exist_ok=True)


image = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("input.jpg not found. Please keep input.jpg in the same folder.")
    exit()

image = cv2.resize(image, (512, 512))


# OpenCV Histogram Equalization
equalized_image = cv2.equalizeHist(image)


# Manual Histogram Equalization
hist = np.bincount(image.flatten(), minlength=256)

pdf = hist / image.size
cdf = np.cumsum(pdf)

mapping = np.round(cdf * 255).astype(np.uint8)

manual_equalized = mapping[image]


# Save Images
cv2.imwrite(os.path.join(output_dir, "original.png"), image)
cv2.imwrite(os.path.join(output_dir, "opencv_equalized.png"), equalized_image)
cv2.imwrite(os.path.join(output_dir, "manual_equalized.png"), manual_equalized)


# Display Images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(equalized_image, cmap="gray")
plt.title("OpenCV Equalized")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(manual_equalized, cmap="gray")
plt.title("Manual Equalized")
plt.axis("off")

plt.suptitle("Histogram Equalization")
plt.savefig(os.path.join(output_dir, "equalization_images.png"), dpi=150, bbox_inches="tight")
plt.show()


# Histogram Comparison
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(image.ravel(), bins=256, range=(0, 256))
plt.title("Original Histogram")

plt.subplot(1, 3, 2)
plt.hist(equalized_image.ravel(), bins=256, range=(0, 256))
plt.title("OpenCV Equalized Histogram")

plt.subplot(1, 3, 3)
plt.hist(manual_equalized.ravel(), bins=256, range=(0, 256))
plt.title("Manual Equalized Histogram")

plt.savefig(os.path.join(output_dir, "histogram_comparison.png"), dpi=150, bbox_inches="tight")
plt.show()


print("Lab 3 completed successfully.")
print("Check the lab3_outputs folder.")

