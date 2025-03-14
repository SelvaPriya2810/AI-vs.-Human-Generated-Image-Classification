# **AI vs. Human-Generated Image Classification** 🎨🤖  

This project classifies images as either **AI-generated or human-created** using **deep learning** and **InceptionV3** with **transfer learning**. The model is trained on a dataset of AI-generated and real images, leveraging data augmentation and binary classification techniques.

---

## **🚀 Features**
- **InceptionV3-based Deep Learning Model**  
- **Transfer Learning for Efficient Training**  
- **Data Augmentation for Generalization**  
- **Binary Classification (AI vs. Human Images)**  
- **Evaluation and Visualization of Predictions**  
- **Deployment-ready for Web Apps (Flask/Streamlit)**  

---

## **📂 Dataset**
The dataset contains:
- **Train Images:** 63,960 images  
- **Validation Images:** 15,990 images  
- **Classes:**  
  - `AI` (AI-Generated Images)  
  - `Human` (Human-Created Images)  

📌 **Dataset Source:** [Kaggle](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset)  

---

## **🔧 Installation**
1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/your-username/ai-vs-human-classification.git
cd ai-vs-human-classification
```
2️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

---

## **📊 Exploratory Data Analysis (EDA)**
Before training, the dataset was analyzed to:
- Check **class distribution**  
- Visualize **AI vs. Human-generated images**  
- Analyze **image dimensions & color patterns**  

📌 **Key Observations:**  
- AI-generated images often have distinct textures.
- The dataset is balanced between AI and human-generated images.
- Some variation in image sizes required preprocessing.

---

## **🏗️ Model Architecture**
- **Pretrained InceptionV3** model with **ImageNet weights**.
- **Global Average Pooling** for feature extraction.
- **Fully Connected Layers** for classification.
- **Binary Crossentropy Loss** and **Adam Optimizer**.

The model was trained for **10 epochs** with a batch size of **32**.

---

## **🎯 Model Training & Evaluation**
- **Training Accuracy:** Achieved **high performance** with proper augmentation.
- **Validation Accuracy:** Maintained consistency without overfitting.
- **Test Accuracy:** Evaluated on unseen data to measure generalization.

📌 **Results:**  
- The model successfully distinguishes AI-generated images from real ones.
- Accuracy can be improved with further fine-tuning and hyperparameter tuning.

---

## **📌 Predictions**
Once trained, the model can classify new images as **AI-generated or human-created** with high accuracy.  

To use the model for predictions:
```python
model.predict(image)
```

---

## **🚀 Deployment**
The model can be deployed using **Flask** or **Streamlit** for an interactive web interface where users can upload images for classification.

---

## **📌 Next Steps**
- **Fine-tuning for better accuracy.**  
- **Model explainability with SHAP/LIME.**  
- **Deploying as a web application.**  

---

## **📝 Author**
Developed by **Selvapriya Selvakumar** 🚀  
