# 🌾 Rahat - The Relief for Farmers

**Rahat** is an AI-based platform built to help farmers with crop recommendations, fertilizer suggestions, and plant disease detection. It uses machine learning to support better decision-making and improve crop yield.

---

## 💡 Motivation

Being from a farmer’s family, I’ve seen closely how agriculture is not just a livelihood but a lifeline for millions, especially in a country like India. Still, many farmers don’t have access to proper tools or scientific support to improve their farming decisions. Through this project, I aim to use Machine Learning to bridge this gap and provide smart, accessible solutions that can truly help on the ground.

Rahat provides three core applications:

- **🌱 Crop Recommendation**  
  Predict the most suitable crop to grow based on soil nutrients and weather conditions.
  
- **💊 Fertilizer Recommendation**  
  Suggest the right fertilizers by analyzing nutrient deficiencies or excesses in the soil.

- **🦠 Plant Disease Detection**  
  Upload a plant leaf image to detect diseases and receive suggestions for treatment.

---

## 📊 Data Sources

- Custom-built **Crop Recommendation** dataset  
- Custom-built **Fertilizer Suggestion** dataset  
- **Plant Disease** dataset (image-based)

---

## 📁 Notebooks

Model training, data analysis, and preprocessing are performed using Jupyter notebooks available in the `notebooks/` directory.

---

## ⚙️ Built With

- Python  
- Flask  
- HTML/CSS & Bootstrap  
- scikit-learn, TensorFlow, OpenCV  

---

## 💻 How to Use

### 🌾 Crop Recommendation System
Enter NPK (Nitrogen-Phosphorus-Potassium) values, state, and city. The system will predict the best crop for that soil and climate.  

### 🧪 Fertilizer Suggestion System  
Enter the soil’s nutrient values and crop name. The system suggests which nutrients are deficient or in excess and recommends suitable fertilizers.

### 🖼️ Plant Disease Detection System  
Upload an image of a plant leaf. The model predicts whether it’s healthy or diseased, and provides details along with remedies.


---

## 📸 Screenshots

### 🏠 Home Page  
![Home Page](https://github.com/ShivtejRakhunde/rahat/blob/main/screenshots/home_page.jpeg)

### 🌱 Crop Recommendation  
![Crop Recommendation](https://github.com/ShivtejRakhunde/rahat/blob/main/screenshots/crop_rec.jpeg)

![Output](https://github.com/ShivtejRakhunde/rahat/blob/main/screenshots/crop_rec_output.jpeg)

### 🧪 Fertilizer Suggestion  
![Fertilizer Suggestion](https://github.com/ShivtejRakhunde/rahat/blob/main/screenshots/fertilizer.jpeg)

![Output](https://github.com/ShivtejRakhunde/rahat/blob/main/screenshots/fertilizer_output.jpeg)

### 🌿 Disease Detection  
![Disease Detection](https://github.com/ShivtejRakhunde/rahat/blob/main/screenshots/disease.jpeg)

![Output](https://github.com/ShivtejRakhunde/rahat/blob/main/screenshots/disease_output.jpeg)

---

## 📈 Future Improvements

There’s still a lot I’d love to work on to make **Rahat** even better:

- 🌾 **Support more crops** for disease detection to cover a wider variety of Indian agriculture.
- 🎯 **Improve prediction accuracy** by training on larger and more diverse datasets.
- 🧪 **Better fertilizer insights**, including organic alternatives and region-specific advice.
- 🌐 **Multilingual support** so farmers can use the app in their local language.
- 📱 **Build a mobile app version** to make it even more accessible on the field.
- 🧠 **Add voice assistant feature** so farmers can talk to the system instead of typing.
- 🔍 **Weather integration** to give real-time advice based on upcoming weather patterns.
