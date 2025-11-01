import os
import cv2
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
import json
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import albumentations as A
from google.colab import files
import urllib.request
import zipfile

def download_sample_dataset():
    base_dir = '/content/weed_detection'
    raw_dir = f'{base_dir}/raw_images'
    processed_dir = f'{base_dir}/processed_images'
    annotations_dir = f'{base_dir}/annotations'
    
    for dir_path in [base_dir, raw_dir, processed_dir, annotations_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    return base_dir, raw_dir, processed_dir, annotations_dir

class WeedImagePreprocessor:    
    def __init__(self, target_size=(640, 640), enhance=True):
        self.target_size = target_size
        self.enhance = enhance
        self.augmentation_pipeline = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
        ])
    
    def preprocess_image(self, image_path, augment=False):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if augment:
            augmented = self.augmentation_pipeline(image=img)
            img = augmented['image']
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        if self.enhance:
            img = self._enhance_image(img)
        return img
    
    def _enhance_image(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        return enhanced
    
    def normalize_for_yolo(self, img):
        return img.astype(np.uint8)
    
    def normalize_for_cnn(self, img):
        return img.astype(np.float32) / 255.0

class YOLOAnnotationGenerator:
    
    def __init__(self, classes=['crop', 'weed', 'background']):
        self.classes = classes
        self.class_to_id = {cls: idx for idx, cls in enumerate(classes)}
    
    def create_annotation(self, image_path, bboxes, class_labels):
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        annotations = []
        for bbox, label in zip(bboxes, class_labels):
            x, y, w, h = bbox
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            norm_width = w / img_width
            norm_height = h / img_height
            class_id = self.class_to_id.get(label, 1)
            annotation_line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
            annotations.append(annotation_line)
        return '\n'.join(annotations)
    
    def save_annotation(self, annotation_text, output_path):
        with open(output_path, 'w') as f:
            f.write(annotation_text)
    
    def generate_classes_file(self, output_path):
        with open(output_path, 'w') as f:
            for cls in self.classes:
                f.write(f"{cls}\n")

class CNNFeatureExtractor:    
    def __init__(self):
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    def extract_features(self, image_path, img_size=(224, 224)):
        img = image.load_img(image_path, target_size=img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = self.model.predict(img_array, verbose=0)
        return features.flatten()
    
    def extract_region_features(self, image_path, bbox, img_size=(224, 224)):
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, y, w, h = bbox
        roi = img[y:y+h, x:x+w]
        temp_path = '/tmp/temp_roi.jpg'
        cv2.imwrite(temp_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
        features = self.extract_features(temp_path, img_size)
        return features

class WeedDetectionPipeline:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / 'raw_images'
        self.processed_dir = self.base_dir / 'processed_images'
        self.annotations_dir = self.base_dir / 'annotations'
        
        self.preprocessor = WeedImagePreprocessor()
        self.annotator = YOLOAnnotationGenerator()
        self.feature_extractor = CNNFeatureExtractor()
        
        self.dataset_info = []
    
    def process_dataset(self, image_paths, generate_synthetic_annotations=True):
        for idx, img_path in enumerate(image_paths):
            try:
                img_name = Path(img_path).stem
                print(f"Processing {idx+1}/{len(image_paths)}: {img_name}")
                processed_img = self.preprocessor.preprocess_image(img_path)
                output_img_path = self.processed_dir / f"{img_name}.jpg"
                cv2.imwrite(str(output_img_path), cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
                if generate_synthetic_annotations:
                    bboxes, labels = self._generate_synthetic_annotations(processed_img.shape)
                else:
                    bboxes, labels = self._load_annotations(img_path)
                annotation_text = self.annotator.create_annotation(
                    output_img_path, bboxes, labels
                )
                annotation_path = self.annotations_dir / f"{img_name}.txt"
                self.annotator.save_annotation(annotation_text, annotation_path)
                global_features = self.feature_extractor.extract_features(str(output_img_path))
                region_features_list = []
                for bbox in bboxes:
                    h, w = processed_img.shape[:2]
                    x = int(bbox[0])
                    y = int(bbox[1])
                    box_w = int(bbox[2])
                    box_h = int(bbox[3])
                    x = max(0, min(x, w - 1))
                    y = max(0, min(y, h - 1))
                    box_w = min(box_w, w - x)
                    box_h = min(box_h, h - y)
                    
                    if box_w > 0 and box_h > 0:
                        region_features = self.feature_extractor.extract_region_features(
                            str(output_img_path), (x, y, box_w, box_h)
                        )
                        region_features_list.append(region_features)
                self.dataset_info.append({
                    'image_filename': f"{img_name}.jpg",
                    'annotation_filename': f"{img_name}.txt",
                    'num_objects': len(bboxes),
                    'class_labels': ','.join(labels),
                    'global_features': global_features.tolist(),
                    'num_regions': len(region_features_list),
                    'image_shape': f"{processed_img.shape[0]}x{processed_img.shape[1]}"
                })
                
            except Exception as e:
                print(f"  ✗ Error processing {img_path}: {str(e)}")
                continue
        
        print("\n✓ Dataset processing complete!\n")
    
    def _generate_synthetic_annotations(self, img_shape):
        h, w = img_shape[:2]
        num_boxes = np.random.randint(1, 4)
        bboxes = []
        labels = []
        
        for _ in range(num_boxes):
            x = np.random.randint(0, w - 100)
            y = np.random.randint(0, h - 100)
            box_w = np.random.randint(50, 200)
            box_h = np.random.randint(50, 200)
            box_w = min(box_w, w - x)
            box_h = min(box_h, h - y)
            bboxes.append((x, y, box_w, box_h))
            labels.append(np.random.choice(['crop', 'weed']))
        
        return bboxes, labels
    def create_feature_csv(self, output_path):
        df = pd.DataFrame(self.dataset_info)
        feature_columns = [f'feature_{i}' for i in range(len(self.dataset_info[0]['global_features']))]
        features_df = pd.DataFrame(
            df['global_features'].tolist(),
            columns=feature_columns
        )
        final_df = pd.concat([
            df[['image_filename', 'annotation_filename', 'num_objects', 
                'class_labels', 'num_regions', 'image_shape']],
            features_df
        ], axis=1)

        final_df.to_csv(output_path, index=False)
        return final_df
    
    def train_test_val_split(self, test_size=0.2, val_size=0.1, random_state=42):
        image_files = list(self.processed_dir.glob('*.jpg'))
        image_names = [f.stem for f in image_files]
        train_val, test = train_test_split(
            image_names,
            test_size=test_size,
            random_state=random_state
        )
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=random_state
        )
        split_dirs = {
            'train': self.base_dir / 'train',
            'val': self.base_dir / 'val',
            'test': self.base_dir / 'test'
        }
        
        for split_name, split_dir in split_dirs.items():
            (split_dir / 'images').mkdir(parents=True, exist_ok=True)
            (split_dir / 'labels').mkdir(parents=True, exist_ok=True)
        splits = {'train': train, 'val': val, 'test': test}
        
        for split_name, image_list in splits.items():
            split_dir = split_dirs[split_name]
            
            for img_name in image_list:
                src_img = self.processed_dir / f"{img_name}.jpg"
                dst_img = split_dir / 'images' / f"{img_name}.jpg"
                shutil.copy2(src_img, dst_img)
                src_label = self.annotations_dir / f"{img_name}.txt"
                dst_label = split_dir / 'labels' / f"{img_name}.txt"
                if src_label.exists():
                    shutil.copy2(src_label, dst_label)
        data_yaml = {
            'path': str(self.base_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.annotator.classes),
            'names': self.annotator.classes
        }
        
        yaml_path = self.base_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            for key, value in data_yaml.items():
                if isinstance(value, list):
                    f.write(f"{key}: {value}\n")
                else:
                    f.write(f"{key}: {value}\n")
        split_info_df = pd.DataFrame({
            'image_name': image_names,
            'split': ['train' if x in train else 'val' if x in val else 'test' for x in image_names]
        })
        split_info_df.to_csv(self.base_dir / 'split_info.csv', index=False)
        return train, val, test

def visualize_sample(image_path, annotation_path, classes):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
            annotations = f.readlines()
        for ann in annotations:
            parts = ann.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                color = (0, 255, 0) if classes[int(class_id)] == 'crop' else (255, 0, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = classes[int(class_id)]
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img


def main():
    base_dir, raw_dir, processed_dir, annotations_dir = download_sample_dataset()
    pipeline = WeedDetectionPipeline(base_dir)
    image_paths = list(Path(raw_dir).glob('*.jpg')) + list(Path(raw_dir).glob('*.png'))
    pipeline.process_dataset(image_paths, generate_synthetic_annotations=True)
    classes_path = Path(annotations_dir) / 'classes.txt'
    pipeline.annotator.generate_classes_file(classes_path)
    csv_path = Path(base_dir) / 'weed_detection_features.csv'
    df = pipeline.create_feature_csv(csv_path)
    train, val, test = pipeline.train_test_val_split(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    sample_imgs = list(Path(processed_dir).glob('*.jpg'))[:3]
    fig, axes = plt.subplots(1, min(3, len(sample_imgs)), figsize=(15, 5))
    if len(sample_imgs) == 1:
        axes = [axes]
    
    for idx, img_path in enumerate(sample_imgs):
        ann_path = Path(annotations_dir) / f"{img_path.stem}.txt"
        img_with_boxes = visualize_sample(img_path, ann_path, pipeline.annotator.classes)
        axes[idx].imshow(img_with_boxes)
        axes[idx].set_title(f"{img_path.stem}")
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{base_dir}/sample_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()
    return pipeline, df

if __name__ == "__main__":
    pipeline, df = main()