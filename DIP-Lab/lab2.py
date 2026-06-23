import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


output_dir = "lab2_outputs"
os.makedirs(output_dir, exist_ok=True)


image = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("input.jpg not found. Please keep input.jpg in the same folder.")
    exit()

image = cv2.resize(image, (512, 512))


# Negative Transformation
negative = 255 - image


# Log Transformation
image_float = image.astype(np.float32)
constant = 255 / np.log(1 + np.max(image_float))
log_image = constant * np.log(1 + image_float)
log_image = np.uint8(np.clip(log_image, 0, 255))


# Gamma Correction
gamma_05 = np.uint8(255 * ((image / 255) ** 0.5))
gamma_20 = np.uint8(255 * ((image / 255) ** 2.0))


# Contrast Stretching
minimum = np.min(image)
maximum = np.max(image)

contrast_stretched = (image - minimum) * (255 / (maximum - minimum))
contrast_stretched = np.uint8(contrast_stretched)


# Save Images
cv2.imwrite(os.path.join(output_dir, "original.png"), image)
cv2.imwrite(os.path.join(output_dir, "negative.png"), negative)
cv2.imwrite(os.path.join(output_dir, "log_transformation.png"), log_image)
cv2.imwrite(os.path.join(output_dir, "gamma_05.png"), gamma_05)
cv2.imwrite(os.path.join(output_dir, "gamma_20.png"), gamma_20)
cv2.imwrite(os.path.join(output_dir, "contrast_stretching.png"), contrast_stretched)


# Display Results
images = [
    image,
    negative,
    log_image,
    gamma_05,
    gamma_20,
    contrast_stretched
]

titles = [
    "Original",
    "Negative",
    "Log Transformation",
    "Gamma = 0.5",
    "Gamma = 2.0",
    "Contrast Stretching"
]

plt.figure(figsize=(13, 8))

for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.suptitle("Image Enhancement Using Point Processing")
plt.savefig(os.path.join(output_dir, "lab2_result.png"), dpi=150, bbox_inches="tight")
plt.show()


print("Lab 2 completed successfully.")
print("Check the lab2_outputs folder.")

