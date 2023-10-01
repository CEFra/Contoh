# Impor library yang dibutuhkan
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Persiapan data latih (contoh data XOR)
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_data = np.array([0, 1, 1, 0])

# Membangun model neural network
model = keras.Sequential([
    keras.layers.Dense(units=2, input_dim=2, activation='sigmoid'), # Layer input
    keras.layers.Dense(units=1, activation='sigmoid')              # Layer output
])

# Kompilasi model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Melatih model
model.fit(input_data, output_data, epochs=10000, verbose=0)

# Evaluasi model
loss, accuracy = model.evaluate(input_data, output_data)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Prediksi dengan model
predictions = model.predict(input_data)
print('Hasil prediksi:')
for i in range(len(input_data)):
    print(f'Input: {input_data[i]}, Prediksi: {round(predictions[i][0])}')
