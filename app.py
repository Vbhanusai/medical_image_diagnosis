import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image
import keras
from keras.applications.vgg16 import preprocess_input 
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)
# Define route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    try:
        if request.method == 'POST':
            # Handle image upload and perform prediction
            image_file = request.files['imagefile']

            # Get the selected model from the form data (pneumonia or cataract)
            selected_model = request.form['model']
            if image_file:
                # Open the uploaded image directly from memory
                image = Image.open(image_file)
                
                size_limit=2000
                # Check the dimensions before saving
                if image.width > size_limit or image.height > size_limit:
                    return f"Image exceeds {size_limit}x{size_limit} size limit. Please re-upload a smaller image."

                # Save the image only if it passes the size check
                if not os.path.exists('uploads'):
                    os.makedirs('uploads')
                image_path = os.path.join('uploads', image_file.filename)
                image.save(image_path)

                if selected_model == 'pneumonia':
                    model_dir = 'pneumonia'
                    class_names = ['Normal lung', 'pneumonic lung']  # Pneumonia class labels
                    channel = 'L'
                elif selected_model == 'cataract':
                    model_dir = 'cataract'
                    class_names = ['Normal eye', 'cataractous eye']  # Cataract class labels
                    channel = 'RGB'
                else:
                    return "Invalid model selected."

                # Load the pickled model architecture and weights for the selected model
                architecture_path = os.path.join('models', model_dir, f'model_architecture_{selected_model}.pkl')
                weights_path = os.path.join('models', model_dir, f'model_weights_{selected_model}.pkl')

                with open(architecture_path, 'rb') as f:
                    loaded_model_architecture = pickle.load(f)

                with open(weights_path, 'rb') as f:
                    loaded_model_weights = pickle.load(f)

                # Create the model using the loaded architecture
                loaded_model = tf.keras.models.model_from_json(loaded_model_architecture)

                # Set the loaded weights to the model
                loaded_model.set_weights(loaded_model_weights)

                # Preprocess the image and make predictions
                image = image.convert(channel).resize((256, 256))
                x = np.array(image)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                # Make predictions
                predictions = loaded_model.predict(x)
                if selected_model == 'pneumonia':
                    predicted_class_index = np.argmax(predictions[0])
                else:
                    predicted_class_index = 1 if (predictions[0] > 0.5) else 0

                # Get the predicted class label
                predicted_class_label = class_names[predicted_class_index]

                try:
                    os.remove(image_path)
                except Exception as e:
                    print(f"Error deleting image file: {e}")
                
                # Return the prediction as a response
                return predicted_class_label

    except Exception as e:
        print(f"Error occurred: {e}")

    # Render the home page template
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)