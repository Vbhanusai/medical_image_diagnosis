import os
from flask import Flask, render_template, request
from PIL import Image
from gradio_client import Client, handle_file

app = Flask(__name__)

# Define route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    clientPneumonia = Client("bhanusAI/pneumonia-detection")
    clientCataract = Client("bhanusAI/cataract-detection")
    try:
        if request.method == 'POST':
            # Handle image upload and perform prediction
            image_file = request.files['imagefile']

            # Get the selected model from the form data (pneumonia or cataract)
            selected_model = request.form['model']
            if image_file:
                # Open the uploaded image directly from memory
                image = Image.open(image_file)
                
                size_limit=500
                # Check the dimensions before saving
                if image.width > size_limit or image.height > size_limit:
                    return f"Image exceeds {size_limit}x{size_limit} size limit. Please re-upload a smaller image."

                # Save the image only if it passes the size check
                if not os.path.exists('uploads'):
                    os.makedirs('uploads')
                image_path = os.path.join('uploads', image_file.filename)
                image.save(image_path)

                if selected_model == 'pneumonia':
                    client = clientPneumonia
                elif selected_model == 'cataract':
                    client = clientCataract
                else:
                    return "Invalid model selected."
                
                result = client.predict(img=handle_file(image_path),api_name="/predict")
                try:
                    os.remove(image_path)
                except Exception as e:
                    print(f"Error deleting image file: {e}")
                return result
    except Exception as e:
        print(f"Error occurred: {e}")

    # Render the home page template
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)