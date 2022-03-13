# Manuel utilisateur

Ce manuel est destiné aux personnes qui souhaiteraient utiliser l'onithomate.
Il explique comment installer le code sur une machine, comment utiliser les
différentes commandes existantes et comment configurer les fichiers nécessaires
au bon fonctionnement du modèle.


### Table des matières

I. [Installation](#installation)
II. [Préparation des données](#data)
III. [Entrainement du modèle](#train)
IV. [Test du modèle](#test)
V. [Evaluation du modèle](#metrics)

# I. Installation du framework  <a id='installation'></a>

## Prérequis 

Les seuls prérequis sont au niveau de Python. Pour le bon fonctionnement des
programmes, il faut installer les modules présents dans le fichier
requirements.txt dans un environnement Anaconda, par exemple.

Le respect des versions des modules est nécessaire pour le bon fonctionnement
du réseau. En effet, Python et Tensorflow ne sont pas toujours compatible.

Pour entrainer le modèle en utilisant un gpu, vous devez installer cuda 11.5.2
et cudnn 8.3.2.44 depuis le site de nvidia.


## Instalation du code

* Pour télécharger le code source, il suffit de rentrer la commande suivante,
en vous plaçant dans un répertoire vierge.

```bash
git init
git clone --single-branch --branch master https://github.com/yassinedehbi/Ornithomate
```


## Importation des données

* Pour récupérer les données du projet, vous pouvez vous rendre au gdrive suivant:

https://drive.google.com/drive/folders/1-xc9tx4t7bGVU3BbIs5RFZ3B8lC2TLA8


# II. Préparation des données <a id='data'></a>

Avant de se lancer dans cette partie, il faut télécharger les données du projet
et les mettre dans le même répertoire que le code selon l'arborescence suivante:

```txt
Ornithomate
├── config.py
├── raw_data
│   ├── task_05-01-2021
│   └── ...
├── annotations
└── Ornithotasks-CVAT_task.csv
Ornithomate-master
├── prepare_data.py
├── test.py
├── train_mobileNet.py
└── metrics.py
```

Avant de lancer le script de préparation des données, vous devez modifier
la variable __MY_PATH__ en mettant le path vers le répertoire qui contient le
code et les données.

Maintenant vous pouvez lancer le script:

```bash
py prepare_data.py
```

Le script parse les xmls qui existent dans le dossier annotations et crée
trois fichiers (train_data, val_data et test_data) selon la répartition
présente dans le fichier Ornithotasks-CVAT_task.

Les fichiers générés sont de la forme suivante:   

Chaque image est représentée par plusieurs lignes selon le nombre d'oiseaux dans l'image.
Format de la ligne: `image_file_path,x_min,y_min,x_max,y_max,class_id`.

Voici un exemple:
```
Ornithomate/raw_data/task_05-01-2021/2021-01-05-15-58-14.jpg,885.77,603.82,1918.11,1088.0,0
Ornithomate/raw_data/task_05-01-2021/2021-01-05-15-58-14.jpg,885.77,603.82,1918.11,1088.0,0
Ornithomate/raw_data/task_05-01-2021/2021-01-05-15-58-16.jpg,1127.0,538.2,1900.7,1012.91,0
...
```

# III. Entrainement du modèle  <a id='train'></a>

Avant d'entrainer le modèle, vous devez modifier la variable __MYY_PATH__ 
en mettant le path vers le répertoire qui contient le code et les données.

Pour réentrainer un modèle déjà existant, vous pouvez modifier la ligne 51
du script train_mobileNet.py en changeant load_pretrained à True
et en mettant dans weights_path le path vers le fichier .h5 du modèle.

Vous pouvez aussi modifier le taux d'apprentissage et le nombre d'epochs.

Après avoir fait cette configuration, vous pouvez lancer le script:

```bash
py train_mobileNet.py
```

Ce script prend beaucoup de temps pour générer un modèle bien entrainé.
Les modèles sont sauvegardés en format .h5 dans le répertoire logs de Ornithomate.


# IV. Test du modèle <a id='test'></a>

Pour tester le modèle généré par la partie train, il faut d'abord faire des
modifications sur le script test.py. Il faut changer __MY_PATH__ (ligne: 42)
en mettant le path vers le répertoire qui contient le code et les données,
et changer self.model_path (ligne: 43) en mettant le path vers le fichier .h5
du modèle à tester, et mettre le path du fichier qui contient les données
de test (ligne: 198).

Après avoir fait ces changements, vous pouvez lancer le script:

```bash
py test.py
```

Ce script génère un fichier .txt dans le répertoire Ornithomate qui contient
les données. Les colonnes du fichier sont: le path de l'image, xmin, ymin, 
xmax, ymax, la vrai classe de l'image, la classe prédite par le modèle,
la confidence du modèle.


# V. Evaluation du modèle <a id='metrics'></a>

Avant de lancer les scripts d'évaluation, vous devez faire des modifications.
Dans confusion.py, il faut changer le path vers le fichier des prédictions généré
par la partie test (ligne: 7). Dans metrics.py, il faut modifier le path vers le fichier
des images de test (ligne : 18) et le path vers le fichier des prédictions (ligne: 20).

Le fichier confusion.py print la précision du modèle sur la totalité des classes, et aussi
la matrice de confusion.

```bash
py confusion.py
```

La courbe Précision-Rappel est un bon moyen d'évaluer les performances d'un détecteur d'objets
car la confiance est modifiée en traçant une courbe pour chaque classe d'objets. Un détecteur d'objet
d'une classe particulière est considéré comme bon si sa précision reste élevée à mesure que le rappel
augmente, ce qui signifie que si vous faites varier le seuil de confiance, la précision et le rappel
seront toujours élevés.

Une autre façon de comparer les performances des détecteurs d'objets consiste à calculer l'aire sous
la courbe (AUC) de la courbe Précision x Rappel. Comme les courbes AP sont souvent des courbes en zigzag
qui montent et descendent, comparer différentes courbes (différents détecteurs) dans le même tracé n'est
généralement pas une tâche facile - car les courbes ont tendance à se croiser très fréquemment. C'est pourquoi
la précision moyenne (AP), une mesure numérique, peut également nous aider à comparer différents détecteurs.
En pratique, AP est la précision moyenne sur toutes les valeurs de rappel entre 0 et 1.

Le fichier metrics.py fournit pour chaque classe la précision moyenne (average precision) et
la courbe precision-recall.

```bash
py metrics.py
```