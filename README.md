# Zunair-SHAH-IRM
README - Détection de tumeurs cérébrales avec MobileNetV2
Ce projet utilise un modèle de réseau de neurones MobileNetV2 pour détecter des tumeurs cérébrales à partir d'images médicales. Le modèle est entraîné sur un dataset contenant des images de tumeurs cérébrales classées en différentes catégories. Ce projet illustre les étapes de prétraitement des données, la construction du modèle, l'entraînement, l'évaluation et la visualisation des résultats.

Prérequis
Avant de commencer, assurez-vous que vous avez installé les dépendances nécessaires dans votre environnement Python.

Installez les bibliothèques nécessaires :

pip install tensorflow numpy matplotlib seaborn scikit-learn
Téléchargez les données Kaggle :
Si vous utilisez Google Colab, vous devez sélectionner le fichier kaggle.json (clé API de Kaggle) dans votre répertoire Colab en suivant ces étapes :

Créez un compte sur Kaggle.

Allez dans Mon profil -> Mes paramètres et téléchargez votre fichier kaggle.json.

Dans Google Colab, téléchargez ce fichier en exécutant la commande suivante dans une cellule :

from google.colab import files
files.upload()  # Téléchargez votre fichier kaggle.json ici
Ensuite, placez le fichier kaggle.json dans le répertoire ~/.kaggle/ :

import os
os.makedirs('/root/.kaggle', exist_ok=True)
os.rename('kaggle.json', '/root/.kaggle/kaggle.json')


Une fois cela fait, vous pouvez télécharger le dataset en utilisant la commande suivante :

!kaggle datasets download -d somedataset/brain-tumor-dataset
