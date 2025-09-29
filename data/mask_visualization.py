import cv2
from matplotlib import pyplot as plt

image_path = "./5-class/train_images/0000.png"
mask_path = "./5-class/train_masks/0000.png"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

segmented_image = cv2.imread(mask_path)
segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

segmented_image = cv2.resize(
    segmented_image,
    (image.shape[1], image.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

alpha = 0.6
beta = 1.0 - alpha
gamma = 0

overlaid_image = cv2.addWeighted(image, beta, segmented_image, alpha, gamma)

plt.figure(figsize=(10, 10))
plt.imshow(overlaid_image)
plt.axis("off")
plt.title("Image with RGB Mask Overlay")
plt.show()