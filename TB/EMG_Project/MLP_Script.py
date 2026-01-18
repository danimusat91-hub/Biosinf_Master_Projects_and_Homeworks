import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns


## Constant parameters
k_Class = 3

## Train data
X_train = np.load("./Data_filtered_500ms_exercises_Train.npy")
Y_train = np.load("./Labels_filtered_500ms_exercises_Train.npy")
shape = X_train.shape
data_reshaped = X_train.reshape(-1, shape[-1])
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_reshaped)
X = data_scaled.reshape(shape)
X_train_reshaped = X.reshape(X.shape[0], -1)
X_train = X_train_reshaped


## Val Data
X_val = np.load("./Data_filtered_500ms_exercises_Val.npy")
Y_val = np.load("./Labels_filtered_500ms_exercises_Val.npy")
shape = X_val.shape
data_reshaped = X_val.reshape(-1, shape[-1])
data_scaled = scaler.fit_transform(data_reshaped)
X = data_scaled.reshape(shape)
X_val_reshaped = X.reshape(X.shape[0], -1)
X_val = X_val_reshaped

## Test Data
X_test = np.load("./Data_filtered_500ms_exercises_Test.npy")
Y_test = np.load("./Labels_filtered_500ms_exercises_Test.npy")
shape = X_test.shape
data_reshaped = X_test.reshape(-1, shape[-1])
data_scaled = scaler.fit_transform(data_reshaped)
X = data_scaled.reshape(shape)
X_test_reshaped = X.reshape(X.shape[0], -1)
X_test = X_test_reshaped



model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(16,activation='relu'))
model.add(Dense(k_Class, activation='softmax'))




model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], )


checkpoint = ModelCheckpoint('best_model_8features_3classesarm.keras', monitor='val_loss', save_best_only=True, mode='min')

history = model.fit(X_train, to_categorical(Y_train, k_Class), epochs=20, batch_size=32, validation_data=(X_val,to_categorical(Y_val)), callbacks=[checkpoint])

loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Plotarea pierderii
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotarea acurateței
plt.subplot(1, 2, 2)
plt.plot(accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Afișarea graficelor
plt.tight_layout()
plt.show()

# Evaluarea modelului pe setul de testare
loss, accuracy = model.evaluate(X_test, to_categorical(Y_test))
print(f'Loss on test set: {loss}')
print(f'Accuracy on test set: {accuracy}')

# Salvarea modelului antrenat
# model.save('final_model_8features_3classesarm.keras')

Y_pred = model.predict(X_test)

y_test_classes = np.argmax(to_categorical(Y_test), axis=1)
y_pred_classes = np.argmax(Y_pred, axis=1)

# Calculați raportul de clasificare, care include acuratețea pentru fiecare clasă
report = classification_report(y_test_classes, y_pred_classes)
print(report)

conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

# Vizualizați matricea de confuzie
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('MLP Confusion Matrix')
plt.show()



