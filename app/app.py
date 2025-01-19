# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Harvestify - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]
            
            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)

from flask import send_file
from gtts import gTTS
import os
import re

def clean_html(raw_html):
    """Remove HTML tags from a string."""
    return re.sub(r'<.*?>', '', raw_html)


@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    try:
        title = 'Harvestify - Fertilizer Suggestion'

        # Retrieve form inputs
        crop_name = str(request.form['cropname']).strip()
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])

        # Read the dataset
        file_path = 'Data/fertilizer.csv'
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' not found.", 500
        df = pd.read_csv(file_path)
        df['Crop'] = df['Crop'].str.strip()

        # Validate crop name
        if crop_name.lower() not in df['Crop'].str.lower().values:
            return f"Error: Crop '{crop_name}' not found in dataset.", 400

        # Fetch nutrient requirements
        crop_data = df[df['Crop'].str.lower() == crop_name.lower()].iloc[0]
        nr, pr, kr = crop_data['N'], crop_data['P'], crop_data['K']

        # Calculate nutrient differences
        n = nr - N
        p = pr - P
        k = kr - K
        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_value = temp[max(temp.keys())]

        # Determine nutrient status
        if max_value == "N":
            key = 'NHigh' if n < 0 else "Nlow"
        elif max_value == "P":
            key = 'PHigh' if p < 0 else "Plow"
        else:
            key = 'KHigh' if k < 0 else "Klow"

        # Ensure fertilizer_dic exists and contains the key
        if key not in fertilizer_dic:
            return f"Error: Recommendation key '{key}' not found in dictionary.", 500

        recommendation = fertilizer_dic[key]
        recommendation1 = clean_html(fertilizer_dic[key])
        from flask import Markup

        # Function to split paragraphs with HTML tags
        def split_paragraph_with_html(paragraph):
            import re
            # Using regex to split by <br/> and remove tags
            return re.split(r'<br\s*/?>', paragraph)

        # Example paragraph with HTML content
        recommendation_paragraph = recommendation

        # Split the paragraph by <br/> tags
        recommendation_sentences = split_paragraph_with_html(recommendation_paragraph)

        # Creating the table with HTML Markup
        table_html = f"""
        <table style="
            border-collapse: collapse; 
            width: 60%; 
            margin: 20px auto; 
            text-align: left; 
            background-color: #ffe6f0; 
            color: #333; 
            border: 1px solid #dd99bb; 
            border-radius: 8px;">
        <thead>
            <tr style="background-color: #ffccde;">
            <th style="
                border: 1px solid #dd99bb; 
                padding: 12px; 
                text-align: center; 
                font-weight: bold; 
                color: #333; 
                font-size: 1.2em;">
                Suggestions for Low Nitrogen
            </th>
            </tr>
        </thead>
        <tbody>
        """

        # Adding sentences to the table
        for sentence in recommendation_sentences:
            if sentence.strip():  # Ensuring the sentence isn't empty
                table_html += f"""
                <tr>
                <td style="
                    border: 1px solid #dd99bb; 
                    padding: 12px; 
                    font-weight: bold;">
                    {sentence.strip()}
                </td>
                </tr>
                """

        # Closing the table
        table_html += """
        </tbody>
        </table>
        """

        # Passing the table as a Markup object
        suggestion_table = Markup(table_html)


       

        
        

        # Convert recommendation to audio using gTTS
        tts = gTTS(text=recommendation1, lang='hi')
        audio_path = 'static/fertilizer_recommendation.mp3'
        tts.save(audio_path)

        # Return the audio file along with the result page
        return render_template(
            'fertilizer-result.html', 
            recommendation=Markup(table_html), 
            title=title,
            audio_file=audio_path
        )

    except Exception as e:
        # Log the exception and return an error response
        app.logger.error(f"Error in fert_recommend: {e}")
        return f"An unexpected error occurred: {str(e)}", 500


# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)
            
            prediction_details = disease_dic.get(prediction, "No details available")
            table_html = f"""
<table style="
    border-collapse: collapse; 
    width: 60%; 
    margin: 20px auto; 
    text-align: left; 
    background-color: #ffe6f0; /* Light pink background */
    color: #333; /* Dark gray text for contrast */
    border: 1px solid #dd99bb; /* Soft border color */
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    border-radius: 8px; /* Rounded corners */
    overflow: hidden;">
  <thead>
    <tr style="background-color: #ffccde; /* Slightly darker pink */">
      <th style="
          border: 1px solid #dd99bb; 
          padding: 12px; 
          text-align: center; 
          font-weight: bold; 
          color: #333; 
          font-size: 1.2em;">
        Disease
      </th>
      <th style="
          border: 1px solid #dd99bb; 
          padding: 12px; 
          text-align: center; 
          font-weight: bold; 
          color: #333; 
          font-size: 1.2em;">
        Details
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="
          border: 1px solid #dd99bb; 
          padding: 12px; 
          font-weight: bold;">
        {prediction}
      </td>
      <td style="
          border: 1px solid #dd99bb; 
          padding: 12px; 
          font-weight: bold;">
        {prediction_details}
      </td>
    </tr>
  </tbody>
</table>
"""

            prediction2 = Markup(table_html)
            return render_template('disease-result.html', prediction=prediction2, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


from flask import Flask, request, jsonify


# In-memory data store (will be reset when the server restarts)
articles = []

@app.route('/community', methods=['POST'])
def add_article():
    """Add an article to the in-memory store."""
    data = request.json  # Get JSON data from the request
    articles.append(data)  # Add to in-memory list
    return jsonify({"message": "Article added successfully!", "article": data}), 201




# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
