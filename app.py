import os
from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import Xception, preprocess_input
from pickle import load


# Initialize Flask app
app = Flask(__name__)

# Load tokenizer
tokenizer = load(open("tokenizer.p","rb"))

# Load pre-trained model
model = load_model('models/model_9.h5')

# Load pre-trained Xception model
xception_model = Xception(include_top=False, pooling="avg")

# Define function to preprocess input image
def preprocess_image(image):
    image = image.resize((299,299))
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

# Define function to generate image caption
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = tokenizer.index_word[pred]
        in_text += ' ' + word
        if word == 'end':
            break
    caption = in_text.split()[1:-1]
    caption = ' '.join(caption)
    return caption

# Define route for home page
@app.route('/')
def home():
    return render_template('home.html')

# Define route for image upload and caption generation
@app.route('/', methods=['POST'])
def upload_image():
    # Get input image
    image_file = request.files['image']
    image = Image.open(image_file)
    
    # Preprocess input image
    processed_image = preprocess_image(image)
    
    # Generate caption
    max_length = 32
    caption = generate_caption(model, tokenizer, processed_image, max_length)
    
    # Render HTML response
    return render_template('home.html', caption=caption, image_file=image_file)
