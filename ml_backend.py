from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import matplotlib.pyplot as plt
from keras import Model
from keras.models import load_model
from keras.applications import VGG16
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from PIL import Image
import json
import requests
from io import BytesIO
import js2py

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Define dataset path and model save path
data_path = 'E:/React Course/oceans-enigma/python-backend/Major project dataset'
model_path = 'E:/React Course/oceans-enigma/python-backend/saved_model.keras'
class_indices_path = 'E:/React Course/oceans-enigma/python-backend/class_indices.json'

# Load species data from species_data.js
# with open('E:/React Course/oceans-enigma/python-backend/species_data.json', 'r') as f:
#     species_data = json.load(f)

# Check if the model already exists
if os.path.exists(model_path):
    print("Loading the existing model...")
    model = load_model(model_path)

    # Load class indices
    if os.path.exists(class_indices_path):
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        print("Class indices loaded successfully.")
    else:
        raise FileNotFoundError("Class indices file not found.")
else:
    print("Training a new model...")

    # Load the base model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    # Add custom layers
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(17, activation='softmax')(x)

    # Final model
    model = Model(inputs=base_model.input, outputs=output)

    # Freeze the base layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Data Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest'
    )

    # Data Augmentation for testing/validation
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Load training data
    training_set = train_datagen.flow_from_directory(
        os.path.join(data_path, 'train'),
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )

    # Load test/validation data
    test_set = test_datagen.flow_from_directory(
        os.path.join(data_path, 'test'),
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )

    # Train the model
    model.fit(
        training_set,
        steps_per_epoch=min(len(training_set), 25),
        epochs=25,
        validation_data=test_set,
        validation_steps=min(len(test_set), 25)
    )

    # Save the trained model
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save class indices
    class_indices = training_set.class_indices
    with open(class_indices_path, 'w') as f:
        json.dump(class_indices, f)
    print("Class indices saved successfully.")

# Species information
# species_info = {
#     "whale_shark": "The whale shark (Rhincodon typus) is the largest fish species on Earth and a gentle filter feeder.",
#     "basking_shark": "The basking shark (Cetorhinus maximus) is the second-largest living shark species, known for its filter-feeding behavior and impressive size.",
#     "tiger_shark": "The tiger shark (Galeocerdo cuvier) is a large, apex predator known for its striped body pattern and opportunistic feeding behavior.",
#     "hammerhead_shark": "The hammerhead shark refers to a group of species in the family Sphyrnidae known for their unique hammer-shaped heads (cephalofoils)."
# }


# Function to predict species
# def predict_species(image_path_or_url):
#     try:
#         # Check if the input is a URL
#         if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
#             # Download the image from the URL
#             response = requests.get(image_path_or_url)
#             response.raise_for_status()  # Raise an error for invalid HTTP responses
#             img = Image.open(BytesIO(response.content)).resize((128, 128)).convert("RGB")
#         else:
#             # Load the image from a local file
#             img = load_img(image_path_or_url, target_size=(128, 128))
        
#         # Convert the image to a numpy array
#         img_array = img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = img_array / 255.0

#         # Predict species
#         prediction = model.predict(img_array)
#         species_index = np.argmax(prediction)
#         species_name = list(class_indices.keys())[list(class_indices.values()).index(species_index)]

#         # Display the image and predicted species
#         plt.imshow(img)
#         plt.axis('off')
#         plt.title(f"Predicted: {species_name}")
#         plt.show()

#         # return species_name
#         return species_name, species_info.get(species_name, "No additional information available.")
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return None


def predict_species(image):
    try:
        # Convert the image to a numpy array
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict species
        prediction = model.predict(img_array)
        species_index = np.argmax(prediction)
        species_name = list(class_indices.keys())[list(class_indices.values()).index(species_index)]

        return species_name
        # return species_name, species_info.get(species_name, "No additional information available.")
    except Exception as e:
        return None, f"Error: {str(e)}"

    
# Example usage
# predicted_species = predict_species(os.path.join(data_path, 'test/whale_shark/00000144.jpeg'))
# predicted_species = predict_species('https://images.pexels.com/photos/6530412/pexels-photo-6530412.jpeg?auto=compress&cs=tinysrgb&w=600')
# print(f"Predicted Species: {predicted_species}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Request received")
        print("Files:", request.files)

        # Check for file in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Open the image and preprocess
        image = Image.open(file.stream).resize((128, 128)).convert("RGB")
        
        # Predict species
        species_name = predict_species(image)

        if species_name:
            return jsonify({
                "species": species_name,
            }), 200
        else:
            return jsonify({"error": "Prediction failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
