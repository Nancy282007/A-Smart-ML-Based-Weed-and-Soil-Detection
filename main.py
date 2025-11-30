import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import time
import io
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Weed and Soil Detector",
    page_icon="🌱",
    layout="wide"
)

CLASS_NAMES = {0: "Crop", 1: "Weed"}
CLASS_COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}

SOIL_CLASS_NAMES = ['Alluvial', 'Laterite', 'Black', 'Yellow', 'Arid', 'Mountain', 'Red']

DISEASE_CLASS_NAMES = ['Apple___Apple_scab',
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

@st.cache_resource
def load_models(yolo_path, classifier_path):
    try:
        yolo_model = YOLO(yolo_path)
        classifier_model = load_model(classifier_path)
        return yolo_model, classifier_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

@st.cache_resource
def load_soil_model(soil_model_path):
    try:
        soil_model = tf.keras.models.load_model(soil_model_path)
        return soil_model
    except Exception as e:
        st.error(f"Error loading soil model: {str(e)}")
        return None

@st.cache_resource
def load_disease_model(disease_model_path):
    try:
        disease_model = tf.keras.models.load_model(disease_model_path)
        return disease_model
    except Exception as e:
        st.error(f"Error loading disease model: {str(e)}")
        return None

def detect_objects(image, yolo_model):
    results = yolo_model.predict(source=image, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    return boxes

def classify_crop(crop_img, classifier_model):
    crop_img_resized = crop_img.resize((224, 224))
    image_array = img_to_array(crop_img_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    start_time = time.time()
    bboxPred, labelPred = classifier_model.predict(image_array, verbose=0)
    inference_time = time.time() - start_time
    class_idx = int(np.argmax(labelPred, axis=1)[0])
    confidence = float(np.max(labelPred) * 100)
    class_name = CLASS_NAMES.get(class_idx, f"Class {class_idx}")
    startX, startY, endX, endY = bboxPred[0]
    
    return {
        'class': class_name,
        'class_idx': class_idx,
        'confidence': confidence,
        'bbox_relative': (startX, startY, endX, endY),
        'inference_time': inference_time,
        'all_probabilities': {
            CLASS_NAMES.get(i, f'Class {i}'): float(labelPred[0][i] * 100)
            for i in range(labelPred.shape[1])
        }
    }

def predict_soil(image, soil_model):
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = soil_model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    predicted_class = SOIL_CLASS_NAMES[predicted_class_idx]
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': {name: prob * 100 for name, prob in zip(SOIL_CLASS_NAMES, predictions[0])}
    }

def predict_disease(image, disease_model):
    img = image.resize((256, 256))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])
    prediction = disease_model.predict(input_arr, verbose=0)
    result_index = np.argmax(prediction)
    confidence = prediction[0][result_index] * 100
    predicted_disease = DISEASE_CLASS_NAMES[result_index]
    
    return {
        'predicted_disease': predicted_disease,
        'confidence': confidence,
        'result_index': result_index
    }

def annotate_image(image_cv, bbox, class_name, confidence, class_idx):
    x1, y1, x2, y2 = map(int, bbox)
    color = CLASS_COLORS.get(class_idx, (255, 255, 255))
    cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 3)
    label = f"{class_name}: {confidence:.2f}%"
    (label_width, label_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )
    label_y = max(y1 - 10, label_height + 10)
    cv2.rectangle(
        image_cv,
        (x1, label_y - label_height - 10),
        (x1 + label_width + 10, label_y),
        color,
        -1
    )
    cv2.putText(
        image_cv,
        label,
        (x1 + 5, label_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    
    return image_cv

st.title("🌱 Agricultural AI Detection System")
st.markdown("Upload an image for crop/weed detection, soil classification, or soil disease detection")

with st.sidebar:
    st.header("🎯 Select Detection Mode")
    detection_mode = st.radio(
        "Choose detection type:",
        ["Crop vs Weed Detection", "Soil Classification", "Soil Disease Detection"]
    )
    st.markdown("---")
    st.header("⚙️ Model Configuration")
    if detection_mode == "Crop vs Weed Detection":
        yolo_model_path = st.text_input(
            "YOLO Model Path",
            value="trained_models/best.pt",
            help="Path to your YOLO model file"
        )
        classifier_model_path = st.text_input(
            "Classifier Model Path",
            value="trained_models/vgg16_model_v6.h5",
            help="Path to your VGG16 classifier model"
        )
        
        if st.button("Load Models"):
            with st.spinner("Loading models..."):
                yolo_model, classifier_model = load_models(yolo_model_path, classifier_model_path)
                if yolo_model and classifier_model:
                    st.session_state['yolo_model'] = yolo_model
                    st.session_state['classifier_model'] = classifier_model
                    st.success("✅ Models loaded successfully!")
    
    elif detection_mode == "Soil Classification":
        soil_model_path = st.text_input(
            "Soil Model Path",
            value="trained_models/final1_soil_classification_model.keras",
            help="Path to your soil classification model"
        )
        
        if st.button("Load Soil Model"):
            with st.spinner("Loading soil model..."):
                soil_model = load_soil_model(soil_model_path)
                if soil_model:
                    st.session_state['soil_model'] = soil_model
                    st.success("✅ Soil model loaded successfully!")
    
    elif detection_mode == "Soil Disease Detection":
        disease_model_path = st.text_input(
            "Disease Model Path",
            value="trained_models/trained_model.keras",
            help="Path to your plant disease model"
        )
        
        if st.button("Load Disease Model"):
            with st.spinner("Loading disease model..."):
                disease_model = load_disease_model(disease_model_path)
                if disease_model:
                    st.session_state['disease_model'] = disease_model
                    st.success("✅ Disease model loaded successfully!")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png', 'bmp'],
    help="Upload an image for analysis"
)

if detection_mode == "Crop vs Weed Detection":
    if uploaded_file is not None:
        if 'yolo_model' not in st.session_state or 'classifier_model' not in st.session_state:
            st.warning("⚠️ Please load the models first using the sidebar")
        else:
            image = Image.open(uploaded_file)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            if st.button("🔍 Detect and Classify", type="primary"):
                with st.spinner("Processing image..."):
                    st.info("Step 1: Detecting objects...")
                    boxes = detect_objects(image, st.session_state['yolo_model'])
                    if len(boxes) == 0:
                        st.warning("No objects detected in the image")
                    else:
                        st.success(f"Detected {len(boxes)} object(s)")
                        st.info("Step 2: Classifying detected objects...")
                        results = []
                        annotated_image = image_cv.copy()
                        st.subheader("Detected Objects")
                        cols = st.columns(min(len(boxes), 4))
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = map(int, box[:4])
                            crop_img = image.crop((x1, y1, x2, y2))
                            classification = classify_crop(crop_img, st.session_state['classifier_model'])
                            annotated_image = annotate_image(
                                annotated_image,
                                (x1, y1, x2, y2),
                                classification['class'],
                                classification['confidence'],
                                classification['class_idx']
                            )
                            with cols[i % 4]:
                                st.image(crop_img, caption=f"Detection {i+1}", use_container_width=True)
                                if classification['class'] == "Crop":
                                    st.success(f"✅ {classification['class']}")
                                else:
                                    st.error(f"❌ {classification['class']}")
                                
                                st.metric("Confidence", f"{classification['confidence']:.2f}%")
                                with st.expander("Details"):
                                    st.write(f"**All Probabilities:**")
                                    for cls, prob in classification['all_probabilities'].items():
                                        st.write(f"- {cls}: {prob:.2f}%")
                                    st.write(f"**Inference Time:** {classification['inference_time']:.4f}s")
                            
                            results.append(classification)
                        
                        with col2:
                            st.subheader("Annotated Image")
                            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                            st.image(annotated_image_rgb, use_container_width=True)
                        
                        st.subheader("📊 Summary Statistics")
                        summary_cols = st.columns(4)
                        
                        crop_count = sum(1 for r in results if r['class'] == 'Crop')
                        weed_count = sum(1 for r in results if r['class'] == 'Weed')
                        avg_confidence = np.mean([r['confidence'] for r in results])
                        avg_inference_time = np.mean([r['inference_time'] for r in results])
                        
                        with summary_cols[0]:
                            st.metric("Total Detections", len(results))
                        with summary_cols[1]:
                            st.metric("Crops", crop_count)
                        with summary_cols[2]:
                            st.metric("Weeds", weed_count)
                        with summary_cols[3]:
                            st.metric("Avg Confidence", f"{avg_confidence:.2f}%")
                        
                        st.subheader("💾 Download Results")
                        annotated_pil = Image.fromarray(annotated_image_rgb)
                        buf = io.BytesIO()
                        annotated_pil.save(buf, format='PNG')
                        buf.seek(0)
                        
                        st.download_button(
                            label="Download Annotated Image",
                            data=buf,
                            file_name="annotated_result.png",
                            mime="image/png"
                        )
    else:
        st.info("Upload an image to get started")

elif detection_mode == "Soil Classification":
    if uploaded_file is not None:
        if 'soil_model' not in st.session_state:
            st.warning("⚠️ Please load the soil model first using the sidebar")
        else:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Input Image")
                st.image(image, use_container_width=True)

            if st.button("🌍 Classify Soil Type", type="primary"):
                with st.spinner("Analyzing soil..."):
                    result = predict_soil(image, st.session_state['soil_model'])
                    with col2:
                        st.subheader("Classification Result")
                        color = 'green' if result['confidence'] > 70 else 'orange' if result['confidence'] > 50 else 'red'
                        st.markdown(f"### Predicted Soil Type: **:{color}[{result['predicted_class']}]**")
                        st.metric("Confidence", f"{result['confidence']:.2f}%")
                    
                    st.subheader("📊 All Class Probabilities")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    soil_types = list(result['all_probabilities'].keys())
                    probabilities = list(result['all_probabilities'].values())
                    predicted_idx = soil_types.index(result['predicted_class'])
                    colors = ['green' if i == predicted_idx else 'skyblue' for i in range(len(soil_types))]
                    bars = ax.barh(soil_types, probabilities, color=colors)
                    ax.set_xlabel('Confidence (%)', fontsize=12)
                    ax.set_title('Prediction Confidence for All Classes', fontsize=12, fontweight='bold')
                    ax.set_xlim(0, 100)
                    
                    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                        ax.text(prob + 1, i, f'{prob:.1f}%', va='center', fontsize=10)
                    
                    st.pyplot(fig)
                    
                    with st.expander("📋 Detailed Probabilities"):
                        for name, prob in result['all_probabilities'].items():
                            st.write(f"**{name}**: {prob:.2f}%")
    else:
        st.info("Upload a soil image to classify")
elif detection_mode == "Soil Disease Detection":
    if uploaded_file is not None:
        if 'disease_model' not in st.session_state:
            st.warning("⚠️ Please load the disease model first using the sidebar")
        else:
            image = Image.open(uploaded_file)
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Image")
                st.image(image, use_container_width=True)
            
            if st.button("🔬 Detect Disease", type="primary"):
                with st.spinner("Analyzing plant..."):
                    result = predict_disease(image, st.session_state['disease_model'])
                    with col2:
                        st.subheader("Detection Result")
                        disease_name = result['predicted_disease']
                        is_healthy = 'healthy' in disease_name.lower()
                        
                        if is_healthy:
                            st.success(f"### ✅ {disease_name}")
                        else:
                            st.error(f"### ⚠️ {disease_name}")
                        
                        st.metric("Confidence", f"{result['confidence']:.2f}%")
                        
                        st.markdown("---")
                        st.markdown(f"**Disease Class Index:** {result['result_index']}")
    else:
        st.info("Upload a plant image to detect diseases")

st.markdown("---")
st.markdown("Built with Streamlit")