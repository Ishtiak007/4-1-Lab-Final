import cv2
import matplotlib.pyplot as plt
import os


output_dir = "lab4_outputs"
os.makedirs(output_dir, exist_ok=True)


image = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("input.jpg not found. Please keep input.jpg in the same folder.")
    exit()

image = cv2.resize(image, (512, 512))


# Average Filters
average_3x3 = cv2.blur(image, (3, 3))
average_5x5 = cv2.blur(image, (5, 5))


# Gaussian Filters
gaussian_3x3 = cv2.GaussianBlur(image, (3, 3), 0)
gaussian_5x5 = cv2.GaussianBlur(image, (5, 5), 0)


# Median Filters
median_3x3 = cv2.medianBlur(image, 3)
median_5x5 = cv2.medianBlur(image, 5)


images = [
    image,
    average_3x3,
    average_5x5,
    gaussian_3x3,
    gaussian_5x5,
    median_3x3,
    median_5x5
]

titles = [
    "Original",
    "Average Filter 3x3",
    "Average Filter 5x5",
    "Gaussian Filter 3x3",
    "Gaussian Filter 5x5",
    "Median Filter 3x3",
    "Median Filter 5x5"
]

file_names = [
    "original.png",
    "average_3x3.png",
    "average_5x5.png",
    "gaussian_3x3.png",
    "gaussian_5x5.png",
    "median_3x3.png",
    "median_5x5.png"
]


for name, img in zip(file_names, images):
    cv2.imwrite(os.path.join(output_dir, name), img)


plt.figure(figsize=(14, 8))

for i in range(len(images)):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.suptitle("Smoothing Filters")
plt.savefig(os.path.join(output_dir, "smoothing_filter_result.png"), dpi=150, bbox_inches="tight")
plt.show()


print("Lab 4 completed successfully.")
print("Check the lab4_outputs folder.")

