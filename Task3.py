import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory
from tf.keras import optimizers
from tf.keras import layers
import random
import os

def Training(Epoch, BatchSize, ImageSize):
    train_val_datagen = ImageDataGenerator(validation_split=0.2, # splits the data 
                                   rescale = 1./255, # rescales the image
                                   shear_range = 0.2, # how much its randomly distorted
                                   zoom_range = 0.2, # how much its randomly zoomed in
                                   horizontal_flip = True) #random horizontal flipping

    test_datagen = ImageDataGenerator(rescale = 1./255)

    #the training sets
    training_set = train_val_datagen.flow_from_directory('Images/Training_images/',
                subset='training', #the validation splits
                target_size = (ImageSize, ImageSize), #image size
                batch_size = BatchSize, #batch size
                class_mode = 'categorical') #parameter for encoding

    validation_set = train_val_datagen.flow_from_directory('Images/Training_images/',
                subset='validation', #the validation splits
                target_size = (ImageSize, ImageSize),
                batch_size = BatchSize,
                class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory('Images/Test_images/',
        target_size = (ImageSize, ImageSize),
        batch_size = BatchSize,
        class_mode = 'categorical')
    NUM_CLASSES = 5

    # Create a sequential model with a list of layers
    model = tf.keras.Sequential([
  #first layer + input layer
  layers.Conv2D(48, (3, 3), input_shape = (128, 128, 3), activation="relu"),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),

  #Second layer + input
  layers.Conv2D(48, (3, 3), activation="relu"),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.3),

  #thrid layer + input
  layers.Conv2D(48, (3, 3), activation="relu"),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),

  #output layer
  layers.Flatten(),
  layers.Dense(128, activation="relu"),
  layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    # Compile and train your model as usual
    model.compile(optimizer = optimizers.Adam(learning_rate=0.001), 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])
    history = model.fit(training_set, validation_data = validation_set, epochs = Epoch)


st.title("EDA")
st.markdown("""
<style>
img {
aspect-ratio: 1 / 1;
}
</style>
""", unsafe_allow_html=True)
st.write("Random Images in Columns of Three images:")
col1, col2, col3, col4 ,col5 = st.columns(5)
randomImage1 = random.randint(1, 100)
randomImage2 = random.randint(1, 100)
randomImage3 = random.randint(1, 100)
with col1:
    st.header("Lions")
    st.image('Images/Training_images/Lion/Lion_'+ str(randomImage1) +'.jpg', use_column_width=True)
    st.image('Images/Training_images/Lion/Lion_'+ str(randomImage2) +'.jpg', use_column_width=True)
    st.image('Images/Training_images/Lion/Lion_'+ str(randomImage3) +'.jpg', use_column_width=True)

with col2:
    st.header("Leopards")
    st.image('Images/Training_images/Leopard/Leopard_'+ str(randomImage1) +'.jpg', use_column_width=True)
    st.image('Images/Training_images/Leopard/Leopard_'+ str(randomImage2) +'.jpg', use_column_width=True)
    st.image('Images/Training_images/Leopard/Leopard_'+ str(randomImage3) +'.jpg', use_column_width=True)

with col3:
    st.header("Cheetahs")
    st.image('Images/Training_images/Cheetah/Cheetah_'+ str(randomImage1) +'.jpg', use_column_width=True)
    st.image('Images/Training_images/Cheetah/Cheetah_'+ str(randomImage2) +'.jpg', use_column_width=True)
    st.image('Images/Training_images/Cheetah/Cheetah_'+ str(randomImage3) +'.jpg', use_column_width=True)

with col4:
    st.header("Lynx's")
    st.image('Images/Training_images/Lynx/Lynx_'+ str(randomImage1) +'.jpg', use_column_width=True)
    st.image('Images/Training_images/Lynx/Lynx_'+ str(randomImage2) +'.jpg', use_column_width=True)
    st.image('Images/Training_images/Lynx/Lynx_'+ str(randomImage3) +'.jpg', use_column_width=True)

with col5:
    st.header("Tigers")
    st.image('Images/Training_images/Tiger/Tiger_'+ str(randomImage1) +'.jpg', use_column_width=True)
    st.image('Images/Training_images/Tiger/Tiger_'+ str(randomImage2) +'.jpg', use_column_width=True)
    st.image('Images/Training_images/Tiger/Tiger_'+ str(randomImage3) +'.jpg', use_column_width=True)


st.title("Model Training")

Epoch = st.slider(label='Epochs', min_value=1, max_value=10, value=5)
BatchSize = st.slider(label='Batch Size', min_value=8, max_value=64, step=8, value=32)
ImageSize = st.number_input(label='Image Size', min_value=16, max_value=256, value=32, step=16)
if st.button("Train Modal"):
    Training(Epoch, BatchSize, ImageSize)


st.title("Visualisation")
