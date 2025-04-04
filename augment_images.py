import albumentations as A
import cv2
import os
from PIL import Image

# Set your input and output directories
input_dir = 'E:/React Course/oceans-enigma/python-backend/Major project dataset/train/whale_shark'
output_dir = 'E:/React Course/oceans-enigma/python-backend/Major project dataset/train/whale_shark'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define augmentations
transform = A.Compose([
    A.Rotate(limit=40, p=0.9),
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
])

# Function to fix PNG profile issues
def fix_png(filepath):
    try:
        img = Image.open(filepath)
        img.save(filepath)
    except Exception as e:
        print(f"Warning: Error fixing PNG profile for {filepath}: {e}")

# Function to generate and save augmented images
def generate_augmented_images_albumentations(image_path, num_augmented_images=5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #albumentations works with RGB.

    for i in range(num_augmented_images):
        augmented = transform(image=image)
        augmented_image = augmented['image']
        output_path = os.path.join(output_dir, f'augmented_{os.path.basename(image_path).split(".")[0]}_{i}.jpg')
        cv2.imwrite(output_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

# Example usage
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        generate_augmented_images_albumentations(image_path)

print("Augmented images generated and saved.")