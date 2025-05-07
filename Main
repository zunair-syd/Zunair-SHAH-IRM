# 1. IMPORTATIONS
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

# 2. PRÉPARATION DES DONNÉES
train_dir = "brain_tumor_dataset/Training"
test_dir = "brain_tumor_dataset/Testing"

img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, shear_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir, target_size=(img_size,img_size), batch_size=batch_size, class_mode='categorical')
test_data = test_datagen.flow_from_directory(test_dir, target_size=(img_size,img_size), batch_size=batch_size, class_mode='categorical', shuffle=False)

# 3. VISUALISATION : DISTRIBUTION DES CLASSES
class_counts = {}
for class_name in os.listdir(train_dir):
    class_folder = os.path.join(train_dir, class_name)
    if os.path.isdir(class_folder):
        class_counts[class_name] = len(os.listdir(class_folder))

plt.figure(figsize=(8,6))
plt.bar(class_counts.keys(), class_counts.values(), color='coral')
plt.title('Répartition des images par classe (train)')
plt.xlabel('Classe')
plt.ylabel("Nombre d'images")
plt.xticks(rotation=15)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 4. MODÈLE
base_model = MobileNetV2(include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 5. ENTRAÎNEMENT
history = model.fit(train_data, epochs=15, validation_data=test_data)

# 6. VISUALISATION : COURBES D'APPRENTISSAGE
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Courbe de précision')
plt.xlabel('Épochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Courbe de perte')
plt.xlabel('Épochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 7. ÉVALUATION
pred = model.predict(test_data)
y_pred = np.argmax(pred, axis=1)
y_true = test_data.classes
class_names = list(test_data.class_indices.keys())

# MATRICE DE CONFUSION
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Prédiction')
plt.ylabel('Réalité')
plt.title('Matrice de confusion')
plt.tight_layout()
plt.show()

# RAPPORT DE CLASSIFICATION
print("=== Rapport de classification ===")
print(classification_report(y_true, y_pred, target_names=class_names))

# 8. AFFICHAGE QUELQUES IMAGES PRÉDITES
import random
from tensorflow.keras.preprocessing import image

plt.figure(figsize=(12, 8))
for i in range(9):
    idx = random.randint(0, len(test_data.filenames) - 1)
    img_path = test_dir + '/' + test_data.filenames[idx]
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)/255.0
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    pred_class = class_names[np.argmax(prediction)]

    plt.subplot(3, 3, i+1)
    plt.imshow(img)
