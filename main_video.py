import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA


def extract_frames_from_video(video_path):
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Error opening video file: {video_path}")
        return frames
    success, image = vidcap.read()
    while success:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        frames.append(resized)
        success, image = vidcap.read()
    return frames

# Function to preprocess the data
def preprocess_data(path_to_dataset, save_path):
    data = []
    labels = []

    # Extract frames and emotion labels from all videos
    for video_speech_actor_folder in os.listdir(path_to_dataset):
        video_speech_actor_folder_path = os.path.join(path_to_dataset, video_speech_actor_folder)
        for actor_folder in os.listdir(video_speech_actor_folder_path):
            actor_folder_path = os.path.join(video_speech_actor_folder_path, actor_folder)
            for subfolder in os.listdir(actor_folder_path):
                subfolder_path = os.path.join(actor_folder_path, subfolder)
                for video_file in os.listdir(subfolder_path):
                    video_path = os.path.join(subfolder_path, video_file)
                    frames = extract_frames_from_video(video_path)
                    emotion_label = int(video_file[6:8]) - 1
                    for frame in frames:
                        data.append(frame)
                        labels.append(emotion_label)

    # Convert data and labels to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Split the data into training and validation sets
    train_data, validation_data, train_labels, validation_labels = train_test_split(data, labels, test_size=0.2)

    # Add an extra dimension to represent grayscale
    train_data = np.expand_dims(train_data, -1)
    validation_data = np.expand_dims(validation_data, -1)

    # Save the preprocessed data
    with open(save_path, 'wb') as f:
        pickle.dump((train_data, validation_data, train_labels, validation_labels), f)

    return train_data, validation_data, train_labels, validation_labels

# Load and preprocess the data
path_to_dataset = "/kaggle/input/ravdess-emotional-speech-video"
save_path = "/kaggle/working/preprocessed_data.pkl"

# Check if the preprocessed data file exists
if os.path.exists(save_path):
    # Load the preprocessed data
    with open(save_path, 'rb') as f:
        train_data, validation_data, train_labels, validation_labels = pickle.load(f)
else:
    # Preprocess the data and save it
    train_data, validation_data, train_labels, validation_labels = preprocess_data(path_to_dataset, save_path)

# Initialize parameters
num_classes = 8 # for 8 different emotions in RAVDESS dataset
batch_size = 64
epochs = 4

# Define emotion labels
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Create the model
model = Sequential()

model.add(Input(shape=(48,48,1)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.00005),metrics=['accuracy'])

# Print values
print(train_data.shape)
print(validation_data.shape)
print(np.any(np.isnan(train_data)))
print(np.any(np.isnan(validation_data)))

# One-hot encode the labels
train_labels = to_categorical(train_labels, num_classes=num_classes)
validation_labels = to_categorical(validation_labels, num_classes=num_classes)

# Train the model
model_info = model.fit(
    train_data,
    train_labels,
    steps_per_epoch=len(train_data) // batch_size,
    epochs=epochs,
    validation_data=(validation_data, validation_labels),
    validation_steps=len(validation_data) // batch_size)

# Predict the values from the validation dataset
Y_pred = model.predict(validation_data)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred, axis=1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(validation_labels, axis=1)

# Calculate the accuracy of our model
accuracy = accuracy_score(y_true=Y_true, y_pred=Y_pred_classes)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))

# F1 score
f1 = f1_score(Y_true, Y_pred_classes, average='weighted')
print("F1 Score: {:.2f}".format(f1))

# Print the confusion matrix
cm = confusion_matrix(y_true=Y_true, y_pred=Y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=emotion_labels)
disp.plot(cmap='viridis', xticks_rotation='vertical')

# Create a dataframe with data
df = pd.DataFrame({'Emotion': Y_pred_classes, 'True Emotion': Y_true})

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Emotion')
plt.ylabel('True Emotion')
plt.show()

# Binarize the labels
Y_true_bin = label_binarize(Y_true, classes=[0, 1, 2, 3, 4, 5, 6, 7])
Y_pred_bin = label_binarize(Y_pred_classes, classes=[0, 1, 2, 3, 4, 5, 6, 7])

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(model_info.history['accuracy'])
plt.plot(model_info.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(model_info.history['loss'])
plt.plot(model_info.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_true_bin[:, i], Y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
for i, label in enumerate(emotion_labels):
    plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(label, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Get the output of the second last layer of the model
model_extract = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)

last_layer_output = model_extract.predict(validation_data)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(last_layer_output)

# Plot the result
plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=Y_true, cmap='viridis', alpha=0.5)
plt.colorbar(label='True Emotion')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Classification Graph')
plt.show()
