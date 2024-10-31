import cv2
import numpy as np

def load_facial_images(image_dir):
    images = []
    labels = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
            img = cv2.resize(img, (48, 48))  # Resize to 48x48 pixels
            img = img.astype('float32') / 255.0  # Normalize pixel values
            images.append(img)
            # Extract label from filename or separate metadata
            label = 1 if 'dementia' in filename else 0
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    images = np.expand_dims(images, -1)  # Add channel dimension
    return images, labels

# Example usage:
facial_images, facial_labels = load_facial_images('path_to_facial_images')