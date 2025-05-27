import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Silencia warnings de TensorFlow

import matplotlib.pyplot as plt
import tensorflow as tf

TAMANO_IMG = 128
base_dir = 'Flowers299'

# Detectar clases
mi_clases = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
print(f"Clases detectadas: {mi_clases}")

# Mapeo de clases a índices
class_to_index = {mi_clase: i for i, mi_clase in enumerate(mi_clases)}

# Recolectar paths e índices
image_paths = []
labels = []

for mi_clase in mi_clases:
    class_dir = os.path.join(base_dir, mi_clase)
    if not os.path.exists(class_dir):
        print(f"No se encontró la carpeta: {class_dir}")
        continue
    for img_file in os.listdir(class_dir):
        full_path = os.path.join(class_dir, img_file)
        image_paths.append(full_path)
        labels.append(class_to_index[mi_clase])

# Convertir a tensor
image_paths_tensor = tf.convert_to_tensor(image_paths)
labels_tensor = tf.convert_to_tensor(labels)

# Crear dataset
def load_and_preprocess_image(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [TAMANO_IMG, TAMANO_IMG])
    img = img / 255.0
    label = tf.one_hot(label, depth=len(mi_clases))
    return img, label

dataset = tf.data.Dataset.from_tensor_slices((image_paths_tensor, labels_tensor))
dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(buffer_size=len(image_paths))

# Dividir en train/test
split_index = int(0.8 * len(image_paths))
train_dataset = dataset.take(split_index).batch(32).prefetch(tf.data.AUTOTUNE)
val_dataset = dataset.skip(split_index).batch(32).prefetch(tf.data.AUTOTUNE)

# Crear modelo
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(TAMANO_IMG, TAMANO_IMG, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(len(mi_clases), activation='softmax')
])

modelo.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Entrenar modelo
print("Entrenando modelo...")
epocas = 60
history = modelo.fit(
    train_dataset,
    epochs=epocas,
    validation_data=val_dataset
)

# Evaluar
test_loss, test_accuracy = modelo.evaluate(val_dataset)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Guardar modelo
export_dir = 'faces-model/1/'
os.makedirs(export_dir, exist_ok=True)
tf.saved_model.save(modelo, export_dir)

# Guardar nombres de clases
with open(os.path.join(export_dir, 'class_names.txt'), 'w') as f:
    for cls in mi_clases:
        f.write(f"{cls}\n")

# Ver estructura exportada
print("Verificando estructura del modelo:")
for root, dirs, files in os.walk(export_dir):
    print(root)
    for file in files:
        print(f"  - {file}")

# Graficar
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
