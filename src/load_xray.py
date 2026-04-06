import cv2
import matplotlib.pyplot as plt

img = cv2.imread("data/sample_xray.jpg", cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap="gray")
plt.title("Chest X-ray")
plt.axis("off")
plt.show()

