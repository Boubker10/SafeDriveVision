# SafeDriveVisionV2

## Description

SafeDriveVisionV2 est un script Python utilisant OpenCV et Mediapipe pour surveiller l'attention du conducteur en temps réel. Le script détecte divers comportements tels que l'utilisation du téléphone portable, le sourire, la fermeture des yeux, et l'orientation de la tête. Il affiche les résultats dans deux fenêtres : une pour la vidéo et une autre pour les informations détectées.


![CAPTURE](https://github.com/user-attachments/assets/e50d2e6f-4eda-4b9f-aafd-f09b3056c06f)

![capcp](https://github.com/user-attachments/assets/12eea00f-e525-4521-96de-020754bc6092)

![capcap](https://github.com/user-attachments/assets/31237958-7a80-481e-a269-ba9a3959ce56)




## Prérequis

- Python 3.6+
- OpenCV
- Mediapipe
- imutils
- NumPy
- argparse

## Installation

Assurez-vous d'avoir Python installé sur votre machine. Ensuite, installez les bibliothèques nécessaires en utilisant pip :

```bash
pip install opencv-python mediapipe imutils numpy

```
# Utilisation

1. Clonez ce dépôt ou copiez le script Python dans votre environnement local.
2. Téléchargez les fichiers `prototxt` et `model` pour le modèle de détection d'objets (par exemple, MobileNet SSD). Placez-les dans le même répertoire que le script.
3. Exécutez le script en utilisant les arguments nécessaires :

```bash
python inference.py -p pro.txt -m SSD.caffemode

```

# Arguments
--prototxt : Chemin vers le fichier prototxt du modèle.
--model : Chemin vers le fichier de poids du modèle.
--confidence : Niveau de confiance minimum pour filtrer les détections faibles (par défaut : 0.7)

```bash
python inference.py -p pro.txt -m SSD.caffemode
```

# Fonctionnalités
Détection de visage : Utilise le cascadeur Haar pour détecter les visages dans le cadre.
Détection de sourire : Utilise le cascadeur Haar pour détecter les sourires.
Suivi des mains : Utilise Mediapipe pour suivre les mouvements des mains.
Mesh du visage : Utilise Mediapipe pour obtenir les points de repère du visage et calculer l'orientation de la tête.
Détection d'utilisation de téléphone portable : Utilise un modèle de détection d'objets pour détecter les téléphones portables.
Affichage des informations : Affiche les informations détectées dans une fenêtre séparée pour une meilleure visibilité.
# Explication du Code
Initialisation des modules : Le script initialise les modules nécessaires, notamment OpenCV, Mediapipe et imutils.
Définition des points de modèle 3D : Les points de repère du visage sont définis pour calculer l'orientation de la tête.
Détection et affichage : Le script capture les images de la webcam, détecte les objets et comportements d'intérêt, et affiche les résultats dans deux fenêtres : une pour la vidéo en direct et une autre pour les informations détectées.
# Auteurs
Boubker BENNANI
#Licence
Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

