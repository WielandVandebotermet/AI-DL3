import streamlit as st
from tensorflow import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras import optimizers
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def Training(Epoch, BatchSize, ImageSize):
    labels = ['Cheetah', 'Leopard', 'Lion', 'Lynx', 'Tiger']
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
    model = keras.Sequential([
    #first layer + input layer
    layers.Conv2D(48, (3, 3), input_shape = (ImageSize, ImageSize, 3), activation="relu"),
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
    layers.Dense(ImageSize, activation="relu"),
    layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    # Compile and train your model as usual
    model.compile(optimizer = optimizers.Adam(learning_rate=0.001), 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])
    
    history = model.fit(training_set, validation_data = validation_set, epochs = Epoch)

    # Create a figure and a grid of subplots with a single call
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    # Plot the loss curves on the first subplot
    ax1.plot(history.history['loss'], label='training loss')
    ax1.plot(history.history['val_loss'], label='validation loss')
    ax1.set_title('Loss curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot the accuracy curves on the second subplot
    ax2.plot(history.history['accuracy'], label='training accuracy')
    ax2.plot(history.history['val_accuracy'], label='validation accuracy')
    ax2.set_title('Accuracy curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Show the figure
    st.pyplot(plt)

    st.title("Visualisation")

    #empty lists
    true_labels = []
    predicted_labels = []
    #the lentgh of the test_set for the loop
    steps = len(test_set)

    #creates 2 arrays and one is the values and the other predicts what it would be.
    for i in range(steps):
        x_batch, y_batch = test_set[i]
        true_labels.extend(np.argmax(y_batch, axis=1))
        predicted_labels.extend(np.argmax(model.predict(x_batch), axis=1))

    # Creating confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plotting confusion matrix
    plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plot.plot(cmap='Purples')
    plt.title('Confusion Matrix')
    col1, col2 = st.columns(2)
    with col1:
        st.header("Our own Trained Model")
        st.pyplot(plt)
    with col2:
        st.header("Google's Teachable Machine")
        st.image('Images/TeachableMachine.PNG', use_column_width=True)
    


def get_random_images(directory, num_images):
    # Get a list of all files in the directory
    all_files = os.listdir(directory)

    # Filter only image files (you might need to adjust this based on your file extensions)
    image_files = [file for file in all_files if file.endswith(('.jpg', '.png', '.jpeg'))]

    # Pick random images from the list
    selected_images = random.sample(image_files, num_images)

    return selected_images

st.title("EDA")

st.markdown("""
<style>
img {
aspect-ratio: 1 / 1;
}
</style>
""", unsafe_allow_html=True)

st.write("Images in Columns of Three images:")
col1, col2, col3, col4 ,col5 = st.columns(5)


Lions = get_random_images('Images/Training_images/Lion/', 3)
Leopards = get_random_images('Images/Training_images/Leopard/', 3)
Cheetahs = get_random_images('Images/Training_images/Cheetah/', 3)
Lynxs = get_random_images('Images/Training_images/Lynx/', 3)
Tigers = get_random_images('Images/Training_images/Tiger/', 3)

Lion_lentgh = len(os.listdir('Images/Training_images/Lion/'))
Leopard_lentgh = len(os.listdir('Images/Training_images/Leopard/'))
Cheetah_lentgh = len(os.listdir('Images/Training_images/Cheetah/'))
Lynx_lentgh = len(os.listdir('Images/Training_images/Lynx/'))
Tiger_lentgh = len(os.listdir('Images/Training_images/Tiger/'))

labels = ['Cheetah', 'Leopard', 'Lion', 'Lynx', 'Tiger']
values = [Cheetah_lentgh, Leopard_lentgh, Lion_lentgh, Lynx_lentgh, Tiger_lentgh]


with col1:
    st.header("Lions")
    st.image('Images/Training_images/Lion/' + Lions[0], use_column_width=True)
    st.image('Images/Training_images/Lion/' + Lions[1], use_column_width=True)
    st.image('Images/Training_images/Lion/' + Lions[2], use_column_width=True)

with col2:
    st.header("Leopards")
    st.image('Images/Training_images/Leopard/'+ Leopards[0], use_column_width=True)
    st.image('Images/Training_images/Leopard/'+ Leopards[1], use_column_width=True)
    st.image('Images/Training_images/Leopard/'+ Leopards[2], use_column_width=True)

with col3:
    st.header("Cheetahs")
    st.image('Images/Training_images/Cheetah/'+ Cheetahs[0], use_column_width=True)
    st.image('Images/Training_images/Cheetah/'+ Cheetahs[1], use_column_width=True)
    st.image('Images/Training_images/Cheetah/'+ Cheetahs[2], use_column_width=True)

with col4:
    st.header("Lynx's")
    st.image('Images/Training_images/Lynx/'+ Lynxs[0], use_column_width=True)
    st.image('Images/Training_images/Lynx/'+ Lynxs[1], use_column_width=True)
    st.image('Images/Training_images/Lynx/'+ Lynxs[2], use_column_width=True)

with col5:
    st.header("Tigers")
    st.image('Images/Training_images/Tiger/'+ Tigers[0], use_column_width=True)
    st.image('Images/Training_images/Tiger/'+ Tigers[1], use_column_width=True)
    st.image('Images/Training_images/Tiger/'+ Tigers[2], use_column_width=True)

fig, ax = plt.subplots()
ax.bar(labels, values)
st.pyplot(fig)

st.title("Model Training")

Epoch = st.slider(label='Epochs', min_value=1, max_value=10, value=5)
BatchSize = st.slider(label='Batch Size', min_value=8, max_value=64, step=8, value=32)
ImageSize = st.number_input(label='Image Size', min_value=16, max_value=256, value=32, step=16)

if st.button("Train Modal"):
    trained = Training(Epoch, BatchSize, ImageSize)

