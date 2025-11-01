import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
import yaml

class Config:
    DATASET_PATH = '/content/dataset'
    OUTPUT_PATH = '/content/processed_data'
    IMG_SIZE = 640 
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42

class SimplePreprocessor:
    def __init__(self, config):
        self.config = config
        self.class_names = []
        
    def create_directories(self):
        dirs = [
            f"{self.config.OUTPUT_PATH}/images/train",
            f"{self.config.OUTPUT_PATH}/images/val",
            f"{self.config.OUTPUT_PATH}/images/test",
            f"{self.config.OUTPUT_PATH}/labels",
        ]
        
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
    
    def find_images(self):
        image_data = []
        root = Path(self.config.DATASET_PATH)
        
        for img_path in root.rglob('*.png'):
            class_name = img_path.parent.name
            image_data.append({
                'path': str(img_path),
                'filename': img_path.name,
                'class': class_name
            })
        
        for img_path in root.rglob('*.jpg'):
            class_name = img_path.parent.name
            image_data.append({
                'path': str(img_path),
                'filename': img_path.name,
                'class': class_name
            })
        
        self.class_names = sorted(list(set([img['class'] for img in image_data])))
        return image_data
    
    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None
        img_resized = cv2.resize(img, (self.config.IMG_SIZE, self.config.IMG_SIZE))
        img_denoised = cv2.fastNlMeansDenoisingColored(img_resized, None, 10, 10, 7, 21)
        
        return img_denoised
    
    def process_all(self):
        print("\n[1/4] Creating directories...")
        self.create_directories()
        print("\n[2/4] Finding images...")
        image_data = self.find_images()
        
        if len(image_data) == 0:
            print("ERROR: No images found!")
            return
        print("\n[3/4] Splitting into train/val/test...")
        df = pd.DataFrame(image_data)
        
        train_df, temp_df = train_test_split(
            df, 
            test_size=(self.config.VAL_RATIO + self.config.TEST_RATIO),
            random_state=self.config.RANDOM_SEED,
            stratify=df['class']
        )
        val_ratio = self.config.VAL_RATIO / (self.config.VAL_RATIO + self.config.TEST_RATIO)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio),
            random_state=self.config.RANDOM_SEED,
            stratify=temp_df['class']
        )
        print(f"  Train: {len(train_df)} images")
        print(f"  Val:   {len(val_df)} images")
        print(f"  Test:  {len(test_df)} images")

        print("\n[4/4] Processing images...")
        self.save_split(train_df, 'train')
        self.save_split(val_df, 'val')
        self.save_split(test_df, 'test')

        self.save_labels(train_df, val_df, test_df)
        self.create_yolo_config()
        
        print(f"\nOutput: {self.config.OUTPUT_PATH}")
    
    def save_split(self, df, split_name):
        output_dir = f"{self.config.OUTPUT_PATH}/images/{split_name}"
        for class_name in self.class_names:
            os.makedirs(f"{output_dir}/{class_name}", exist_ok=True)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            img = self.preprocess_image(row['path'])
            
            if img is not None:
                output_path = f"{output_dir}/{row['class']}/{row['filename']}"
                cv2.imwrite(output_path, img)
    
    def save_labels(self, train_df, val_df, test_df):        
        train_df[['filename', 'class']].to_csv(
            f"{self.config.OUTPUT_PATH}/labels/train.csv", index=False
        )
        val_df[['filename', 'class']].to_csv(
            f"{self.config.OUTPUT_PATH}/labels/val.csv", index=False
        )
        test_df[['filename', 'class']].to_csv(
            f"{self.config.OUTPUT_PATH}/labels/test.csv", index=False
        )
        with open(f"{self.config.OUTPUT_PATH}/labels/classes.txt", 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
    
    def create_yolo_config(self):        
        yolo_config = {
            'path': self.config.OUTPUT_PATH,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        config_path = f"{self.config.OUTPUT_PATH}/weed_data.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(yolo_config, f)


def visualize_samples(config):
    train_dir = f"{config.OUTPUT_PATH}/images/train"
    samples = []
    for class_name in os.listdir(train_dir):
        class_dir = f"{train_dir}/{class_name}"
        if os.path.isdir(class_dir):
            images = os.listdir(class_dir)
            if images:
                samples.append((class_name, f"{class_dir}/{images[0]}"))
    
    n_samples = min(6, len(samples))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (class_name, img_path) in enumerate(samples[:n_samples]):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f"Class: {class_name}\nSize: {img.shape[0]}x{img.shape[1]}")
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_PATH}/samples.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    config = Config()
    preprocessor = SimplePreprocessor(config)
    preprocessor.process_all()
    visualize_samples(config)