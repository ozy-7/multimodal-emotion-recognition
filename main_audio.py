import glob
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import soundfile as sf
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, BatchNormalization, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from sklearn.manifold import TSNE

def extract_feature(file_name, mfcc, chroma, mel):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs.flatten()))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma.flatten()))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel.flatten())) 
    # Ensure the feature vector has a length of 180 (or any other fixed length)
    result = np.pad(result, (0, 180-len(result)), 'constant', constant_values=(0, 0)) if len(result) < 180 else result[:180]
    return result

# Define emotions dictionary and observed_emotions list
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

observed_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def load_data(test_size=0.2):
    x, y = [], []  # Initialize x and y as empty lists
    files = glob.glob("/kaggle/input/ravdess-audio/Actor_*/*.wav")
    print(f"Found {len(files)} files.")
    for file in files:
        file_name = os.path.basename(file)
        emotion = emotions.get(file_name.split("-")[2])
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    if len(x) == 0 or len(y) == 0:
        print("No data found. Please check the file paths and ensure the dataset is correctly placed.")
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

x_train, x_test, y_train, y_test = load_data(test_size=0.25)

if len(x_train) == 0 or len(x_test) == 0:
    print("No training or testing data was loaded. Please check the dataset and the file paths.")
else:
    print((x_train.shape[0], x_test.shape[0]))
    print(f'Features extracted: {x_train.shape[1]}')
    print(type(x_train), x_train.shape)
    print(type(x_test), x_test.shape)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    le = LabelEncoder()
    y_train = to_categorical(le.fit_transform(y_train))  # One-hot encode y_train
    y_test = to_categorical(le.transform(y_test))  # One-hot encode y_test

    model = Sequential()
    model.add(Input(shape=(x_train.shape[1], 1)))
    model.add(GRU(256, return_sequences=True, kernel_regularizer=regularizers.l2(1e-5)))  # GRU layer
    model.add(GRU(256, return_sequences=True, kernel_regularizer=regularizers.l2(1e-5)))  # GRU layer
    model.add(GRU(128, return_sequences=True, kernel_regularizer=regularizers.l2(1e-5)))  # GRU layer
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation('softmax'))

    # Configures the model for training
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    model.summary()

    # Training the model
    history = model.fit(x_train, y_train, batch_size=256, epochs=400, validation_data=(x_test, y_test), shuffle=True)

    # Plotting the accuracy and loss over the epochs
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

    # Evaluating the model
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true, y_pred_classes)
    f1 = f1_score(y_true, y_pred_classes, average='macro')

    print(f'Accuracy: {accuracy}')
    print(f'F1 Score: {f1}')

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=observed_emotions)
    cm_display.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=observed_emotions))

    # ROC Curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(observed_emotions)):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for each class
    plt.figure()
    for i in range(len(observed_emotions)):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {observed_emotions[i]} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    # t-SNE for visualization
    x_test_2d = TSNE(n_components=2).fit_transform(x_test.reshape(x_test.shape[0], -1))

    # Plot classification graph
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=x_test_2d[:, 0], y=x_test_2d[:, 1], hue=le.inverse_transform(y_pred_classes), style=le.inverse_transform(y_true), palette='deep')
    plt.title('Classification Graph')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

