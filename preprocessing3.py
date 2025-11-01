import os 
import tensorflow as tf                                    
import matplotlib.pyplot as plt                         
import matplotlib.image as mpimg                         
from tensorflow import keras 
from keras.models import Sequential, Model      
from keras.optimizers import Adam               
from keras.callbacks import EarlyStopping      
from keras.regularizers import l1, l2 
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, BatchNormalization  
from keras.applications import DenseNet121, EfficientNetB4, Xception, VGG16, VGG19   

directory = '/kaggle/input/weed-detection/train'

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,        
    width_shift_range=0.2,    
    height_shift_range=0.2,   
    shear_range=0.2,          
    zoom_range=0.2,           
    horizontal_flip=True,     
    fill_mode='nearest'  
)

train_data = datagen.flow_from_directory(
    directory,
    classes=['.'], 
    class_mode='categorical',
    target_size=(256, 256),
    batch_size=32,
)

directory = '/kaggle/input/weed-detection/test'

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_data = datagen.flow_from_directory(
    directory,
    classes=['.'],  
    class_mode='categorical',
    target_size=(256, 256),
    batch_size=32,
)

directory_path = '/kaggle/input/weed-detection/train'
allowed_extensions = ('.bmp', '.gif', '.jpeg', '.jpg', '.png')

file_list = [file_name for file_name in os.listdir(directory_path) if os.path.splitext(file_name)[-1].lower() in allowed_extensions]
for file_name in file_list[:5]: 
    img_path = os.path.join(directory_path, file_name)
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.show()

conv_base = DenseNet121(
    weights='imagenet',
    include_top = False,
    input_shape=(256,256,3),
    pooling='avg'
)
conv_base.trainable = False
conv_base.summary()

model = Sequential()
model.add(conv_base)
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.35))
model.add(BatchNormalization())
model.add(Dense(120, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.summary()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=5,
    verbose=0,
    mode="auto",
    baseline=None, 
    restore_best_weights=False,
)

history = model.fit(train_data, epochs=10, validation_data=test_data, callbacks=[early_stopping])
evaluation = model.evaluate(test_data)
print("Validation Loss:", evaluation[0])
print("Validation Accuracy:", evaluation[1])

def plot_metrics(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

plot_metrics(history)
