import cv2
import numpy as np
import pickle
import os

# Path to your CIFAR-10 batches
cifar10_folder = "./cifar-10-batches-py"
batch_files = [f"data_batch_{i}" for i in range(1, 6)]  # List for data_batch_1 to data_batch_5

# Create output folder for videos
os.makedirs("cifar10_videos", exist_ok=True)

# Helper function to apply transformations to an image
def apply_transformations(image, angle=0, scale=1.0, shift=(0, 0)):
    h, w = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
    rotation_matrix[0, 2] += shift[0]
    rotation_matrix[1, 2] += shift[1]
    return cv2.warpAffine(image, rotation_matrix, (w, h))

# Process each batch file
for batch_file in batch_files:
    # Load the batch file
    with open(os.path.join(cifar10_folder, batch_file), 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
        images = batch[b'data']
        labels = batch[b'labels']

    # Reshape images (assuming CIFAR-10 standard 32x32x3 images)
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Process each class label within the batch
    unique_labels = set(labels)
    for class_label in unique_labels:
        class_images = images[np.array(labels) == class_label]
        video_name = f"cifar10_videos/class_{class_label}_{batch_file}.avi"
        video_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 10, (32, 32))

        for i in range(30):  # Generate 30 frames per class
            img = class_images[np.random.randint(len(class_images))]
            angle = np.random.uniform(-30, 30)
            scale = np.random.uniform(0.8, 1.2)
            shift = (np.random.randint(-5, 5), np.random.randint(-5, 5))
            transformed_img = apply_transformations(img, angle, scale, shift)
            video_writer.write(transformed_img)

        video_writer.release()
        print(f"Saved video for class: {class_label} from {batch_file}")

print("All videos generated!")
