import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import gdown

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation

def vgg_face():
    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model

# Load VGG model
model = vgg_face()

# Download model weights from Google Drive
file_id = '1xzSU8THiNVlX0ECQ2ALPQKbnGMw8zWEv'  # Replace with your actual file ID
destination = 'vgg_face_weights.h5'
gdown.download(f'https://drive.google.com/uc?id={file_id}', destination, quiet=False)

# Load the weights
model.load_weights(destination)

from tensorflow.keras.models import Model
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

# Load the necessary models and label encoder
cnn_model_loaded = load_model('cnn_model.h5')
le = joblib.load('label_encoder_cnn.pkl')

# Function to load and preprocess an image
def load_image(image_file):
    img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = (img / 255.).astype(np.float32)
    img = cv2.resize(img, dsize=(224, 224))
    return img

# Function to predict identity using the CNN model
def predict_image_with_cnn(image):
    embedding_vector = vgg_face_descriptor.predict(np.expand_dims(image, axis=0))[0]
    embedding_vector = embedding_vector.reshape(1, -1)  # Reshape for prediction
    prediction = cnn_model_loaded.predict(embedding_vector)
    predicted_class = np.argmax(prediction, axis=1)
    identity = le.inverse_transform(predicted_class)
    return identity[0]

# Streamlit app layout
st.title("Face Recognition of Celebrity")
st.write("Upload an image of a person to predict their identity.")

# Modal Popup for Instructions
with st.expander("The Evolution and Impact of Celebrity Face Recognition Technology", expanded=False):
    st.write(
        "Face recognition technology, particularly in the context of celebrity identification, has become a powerful tool in various applications, from security to entertainment. By leveraging advanced deep learning models and pretrained architectures, systems can accurately analyze facial features and match them against a vast database of known identities. The integration of machine learning with facial recognition not only enhances user experience by providing quick and accurate results but also opens up creative avenues in the realm of digital art and social media, where filters and effects can further transform and emphasize celebrity images. As technology continues to evolve, the potential for face recognition in recognizing and celebrating public figures only expands, making it an engaging and impactful field."
    )

# Uploading the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = load_image(uploaded_file)
    st.image(image, channels="RGB", caption="Uploaded Image", use_column_width=True)

    # Predict the identity
    predicted_identity = predict_image_with_cnn(image)
    st.write(f'Predicted identity: **{predicted_identity}**')
