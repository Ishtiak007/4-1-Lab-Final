import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


output_dir = "lab6_outputs"
os.makedirs(output_dir, exist_ok=True)


image = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("input.jpg not found. Please keep input.jpg in the same folder.")
    exit()

image = cv2.resize(image, (512, 512))


def add_salt_pepper_noise(img, noise_amount=0.05):
    noisy_img = img.copy()

    total_pixels = img.size
    salt_pixels = int(total_pixels * noise_amount / 2)
    pepper_pixels = int(total_pixels * noise_amount / 2)

    row, col = img.shape

    salt_y = np.random.randint(0, row, salt_pixels)
    salt_x = np.random.randint(0, col, salt_pixels)
    noisy_img[salt_y, salt_x] = 255

    pepper_y = np.random.randint(0, row, pepper_pixels)
    pepper_x = np.random.randint(0, col, pepper_pixels)
    noisy_img[pepper_y, pepper_x] = 0

    return noisy_img


def add_gaussian_noise(img, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, img.shape)
    noisy_img = img.astype(np.float32) + gaussian_noise
    noisy_img = np.clip(noisy_img, 0, 255)

    return np.uint8(noisy_img)


def harmonic_mean_filter(img, kernel_size=3):
    img_float = img.astype(np.float32)

    img_float[img_float == 0] = 0.00001

    inverse = 1.0 / img_float
    average_inverse = cv2.blur(inverse, (kernel_size, kernel_size))
    output = 1.0 / average_inverse

    output = np.clip(output, 0, 255)
    return np.uint8(output)


def calculate_psnr(original, processed):
    mse = np.mean((original.astype(np.float32) - processed.astype(np.float32)) ** 2)

    if mse == 0:
        return float("inf")

    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr


np.random.seed(10)


# Noise Add
salt_pepper = add_salt_pepper_noise(image, noise_amount=0.05)
gaussian_noise_img = add_gaussian_noise(image, mean=0, sigma=25)


# Filters on Salt & Pepper Noise
average_sp = cv2.blur(salt_pepper, (3, 3))
median_sp = cv2.medianBlur(salt_pepper, 3)
harmonic_sp = harmonic_mean_filter(salt_pepper, kernel_size=3)


# Filters on Gaussian Noise
average_gaussian = cv2.blur(gaussian_noise_img, (3, 3))
median_gaussian = cv2.medianBlur(gaussian_noise_img, 3)


images = [
    image,
    salt_pepper,
    average_sp,
    median_sp,
    harmonic_sp,
    gaussian_noise_img,
    average_gaussian,
    median_gaussian
]

titles = [
    "Original",
    "Salt & Pepper Noise",
    "Average Filter on S&P",
    "Median Filter on S&P",
    "Harmonic Mean Filter",
    "Gaussian Noise",
    "Average Filter on Gaussian",
    "Median Filter on Gaussian"
]

file_names = [
    "original.png",
    "salt_pepper_noise.png",
    "average_filter_sp.png",
    "median_filter_sp.png",
    "harmonic_filter_sp.png",
    "gaussian_noise.png",
    "average_filter_gaussian.png",
    "median_filter_gaussian.png"
]


for name, img in zip(file_names, images):
    cv2.imwrite(os.path.join(output_dir, name), img)


plt.figure(figsize=(14, 8))

for i in range(len(images)):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.suptitle("Noise Addition and Image Restoration")
plt.savefig(os.path.join(output_dir, "noise_restoration_result.png"), dpi=150, bbox_inches="tight")
plt.show()


# PSNR Results
print("\nPSNR Results:")
print(f"Salt & Pepper Noise: {calculate_psnr(image, salt_pepper):.2f} dB")
print(f"Average Filter on S&P: {calculate_psnr(image, average_sp):.2f} dB")
print(f"Median Filter on S&P: {calculate_psnr(image, median_sp):.2f} dB")
print(f"Harmonic Mean Filter on S&P: {calculate_psnr(image, harmonic_sp):.2f} dB")
print(f"Gaussian Noise: {calculate_psnr(image, gaussian_noise_img):.2f} dB")
print(f"Average Filter on Gaussian Noise: {calculate_psnr(image, average_gaussian):.2f} dB")
print(f"Median Filter on Gaussian Noise: {calculate_psnr(image, median_gaussian):.2f} dB")


print("\nLab 6 completed successfully.")
print("Check the lab6_outputs folder.")

