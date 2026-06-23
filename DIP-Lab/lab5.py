import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


output_dir = "lab5_outputs"
os.makedirs(output_dir, exist_ok=True)


image = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("input.jpg not found. Please keep input.jpg in the same folder.")
    exit()

image = cv2.resize(image, (512, 512))


# Sobel Edge Detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

sobel_x = np.uint8(np.absolute(sobel_x))
sobel_y = np.uint8(np.absolute(sobel_y))

sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)


# Prewitt Edge Detection
prewitt_x_kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

prewitt_y_kernel = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])

prewitt_x = cv2.filter2D(image, -1, prewitt_x_kernel)
prewitt_y = cv2.filter2D(image, -1, prewitt_y_kernel)

prewitt_combined = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)


# Laplacian Edge Detection
laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))


# Canny Edge Detection
canny = cv2.Canny(image, 100, 200)


images = [
    image,
    sobel_x,
    sobel_y,
    sobel_combined,
    prewitt_combined,
    laplacian,
    canny
]

titles = [
    "Original",
    "Sobel X",
    "Sobel Y",
    "Sobel Combined",
    "Prewitt",
    "Laplacian",
    "Canny"
]

file_names = [
    "original.png",
    "sobel_x.png",
    "sobel_y.png",
    "sobel_combined.png",
    "prewitt.png",
    "laplacian.png",
    "canny.png"
]


for name, img in zip(file_names, images):
    cv2.imwrite(os.path.join(output_dir, name), img)


plt.figure(figsize=(14, 8))

for i in range(len(images)):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.suptitle("Edge Detection Methods")
plt.savefig(os.path.join(output_dir, "edge_detection_result.png"), dpi=150, bbox_inches="tight")
plt.show()


print("Lab 5 completed successfully.")
print("Check the lab5_outputs folder.")

