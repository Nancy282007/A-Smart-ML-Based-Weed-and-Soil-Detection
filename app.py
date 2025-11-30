import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time
import io

st.set_page_config(
    page_title="Weed and Soil Detector",
    page_icon="🌱",
    layout="wide"
)

CLASS_NAMES = {0: "Crop", 1: "Weed"}
CLASS_COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}

@st.cache_resource
def load_models(yolo_path, classifier_path):
    """Load YOLO and classification models"""
    try:
        yolo_model = YOLO(yolo_path)
        classifier_model = load_model(classifier_path)
        return yolo_model, classifier_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def detect_objects(image, yolo_model):
    """Detect objects using YOLO and return bounding boxes"""
    results = yolo_model.predict(source=image, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    return boxes

def classify_crop(crop_img, classifier_model):
    """Classify a cropped image as Crop or Weed"""
    # Resize and preprocess
    crop_img_resized = crop_img.resize((224, 224))
    image_array = img_to_array(crop_img_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Predict
    start_time = time.time()
    bboxPred, labelPred = classifier_model.predict(image_array, verbose=0)
    inference_time = time.time() - start_time
    
    # Get classification results
    class_idx = int(np.argmax(labelPred, axis=1)[0])
    confidence = float(np.max(labelPred) * 100)
    class_name = CLASS_NAMES.get(class_idx, f"Class {class_idx}")
    
    # Get bounding box from classifier (relative coordinates)
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

def annotate_image(image_cv, bbox, class_name, confidence, class_idx):
    """Draw bounding box and label on image"""
    x1, y1, x2, y2 = map(int, bbox)
    color = CLASS_COLORS.get(class_idx, (255, 255, 255))
    
    # Draw rectangle
    cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 3)
    
    # Prepare label
    label = f"{class_name}: {confidence:.2f}%"
    (label_width, label_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )
    
    # Draw label background
    label_y = max(y1 - 10, label_height + 10)
    cv2.rectangle(
        image_cv,
        (x1, label_y - label_height - 10),
        (x1 + label_width + 10, label_y),
        color,
        -1
    )
    
    # Draw label text
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

# Main app
st.title("🌱 Crop vs Weed Detection System")
st.markdown("Upload an image to detect and classify crops and weeds")

# Sidebar for model paths
with st.sidebar:
    st.header("⚙️ Model Configuration")
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

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png', 'bmp'],
    help="Upload an image containing crops and/or weeds"
)

if uploaded_file is not None:
    # Check if models are loaded
    if 'yolo_model' not in st.session_state or 'classifier_model' not in st.session_state:
        st.warning("⚠️ Please load the models first using the sidebar")
    else:
        # Load image
        image = Image.open(uploaded_file)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Display original image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Process button
        if st.button("🔍 Detect and Classify", type="primary"):
            with st.spinner("Processing image..."):
                # Step 1: Detect objects with YOLO
                st.info("Step 1: Detecting objects...")
                boxes = detect_objects(image, st.session_state['yolo_model'])
                
                if len(boxes) == 0:
                    st.warning("No objects detected in the image")
                else:
                    st.success(f"Detected {len(boxes)} object(s)")
                    
                    # Step 2: Classify each detected object
                    st.info("Step 2: Classifying detected objects...")
                    
                    results = []
                    annotated_image = image_cv.copy()
                    
                    # Create columns for cropped images
                    st.subheader("Detected Objects")
                    cols = st.columns(min(len(boxes), 4))
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box[:4])
                        
                        # Crop image
                        crop_img = image.crop((x1, y1, x2, y2))
                        
                        # Classify crop
                        classification = classify_crop(crop_img, st.session_state['classifier_model'])
                        
                        # Annotate original image
                        annotated_image = annotate_image(
                            annotated_image,
                            (x1, y1, x2, y2),
                            classification['class'],
                            classification['confidence'],
                            classification['class_idx']
                        )
                        
                        # Display cropped image with classification
                        with cols[i % 4]:
                            st.image(crop_img, caption=f"Detection {i+1}", use_container_width=True)
                            
                            # Classification info
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
                    
                    # Display annotated image
                    with col2:
                        st.subheader("Annotated Image")
                        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                        st.image(annotated_image_rgb, use_container_width=True)
                    
                    # Summary statistics
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
                    
                    # Download button for annotated image
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
    st.info("👆 Upload an image to get started")
    
    # Instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Load Models**: Enter the paths to your YOLO and VGG16 models in the sidebar
        2. **Upload Image**: Click the upload button and select an image
        3. **Detect & Classify**: Click the button to process the image
        4. **View Results**: See detected objects, classifications, and download annotated image
        
        **Model Requirements:**
        - YOLO model for object detection (`.pt` file)
        - VGG16 classifier model for crop/weed classification (`.h5` file)
        """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • YOLO • TensorFlow")