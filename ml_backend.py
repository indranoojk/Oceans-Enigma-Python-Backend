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
CORS(app)

# Defining dataset path and model save path
data_path = 'E:/React Course/oceans-enigma/python-backend/Major project dataset'
model_path = 'E:/React Course/oceans-enigma/python-backend/saved_model.keras'
class_indices_path = 'E:/React Course/oceans-enigma/python-backend/class_indices.json'

# Checking if the model already exists
if os.path.exists(model_path):
    print("Loading the existing model...")
    model = load_model(model_path)

    # Loading class indices
    if os.path.exists(class_indices_path):
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        print("Class indices loaded successfully.")
    else:
        raise FileNotFoundError("Class indices file not found.")
else:
    print("Training a new model...")

    # Loading the base model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    # Adding the custom layers
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(17, activation='softmax')(x)

    # Final model after customization
    model = Model(inputs=base_model.input, outputs=output)

    # Freezing the base layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compiling the model
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

    # Loading training data
    training_set = train_datagen.flow_from_directory(
        os.path.join(data_path, 'train'),
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )

    # Loading test/validation data
    test_set = test_datagen.flow_from_directory(
        os.path.join(data_path, 'test'),
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )

    # Training the model
    model.fit(
        training_set,
        steps_per_epoch=min(len(training_set), 25),
        epochs=25,
        validation_data=test_set,
        validation_steps=min(len(test_set), 25)
    )

    # Saving the trained model for future usage
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Saving class indices to send data easily to frontend
    class_indices = training_set.class_indices
    with open(class_indices_path, 'w') as f:
        json.dump(class_indices, f)
    print("Class indices saved successfully.")

def predict_species(image):
    try:
        # Converting the image to a numpy array
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predicting species
        prediction = model.predict(img_array)
        species_index = np.argmax(prediction)
        species_name = list(class_indices.keys())[list(class_indices.values()).index(species_index)]

        return species_name
        # return species_name, species_info.get(species_name, "No additional information available.")
    except Exception as e:
        return None, f"Error: {str(e)}"

    
# Example usage of our model
# predicted_species = predict_species(os.path.join(data_path, 'test/whale_shark/00000144.jpeg'))
# predicted_species = predict_species('https://images.pexels.com/photos/6530412/pexels-photo-6530412.jpeg?auto=compress&cs=tinysrgb&w=600')
# print(f"Predicted Species: {predicted_species}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Request received")
        print("Files:", request.files)

        # Checking for file in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Opening the image for preprocessing
        image = Image.open(file.stream).resize((128, 128)).convert("RGB")
        
        # Predicting species
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
