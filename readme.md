# Detection de Discours de Haine sur les Réseaux Sociaux

## Introduction

Ce projet vise à détecter les discours de haine en langue française sur les réseaux sociaux. Le discours de haine en ligne est un problème croissant avec des implications sociales et psychologiques significatives. L'objectif est de créer un environnement en ligne plus sûr et plus inclusif en développant des systèmes de détection efficaces.

## Description du Projet

Notre projet se concentre sur le développement d'un outil monolingue spécifiquement pour le français, en abordant le contexte linguistique et culturel unique. Nous avons créé un ensemble de données binaire, sans biais, et adapté à cet objectif. Notre outil intègre l'IA explicable pour justifier ses décisions, offrant transparence et aidant à la compréhension des résultats. Nous avons également développé une application permettant aux utilisateurs d'entrer leurs propres phrases pour tester la présence de discours de haine.

### Principales Contributions

- **Développement d'un ensemble de données personnalisé**: Collecte de tweets en français et génération de phrases artificielles.
- **Prétraitement des données**: Nettoyage des données pour améliorer la qualité, incluant la lemmatisation et la suppression des caractères spéciaux.
- **Formation de modèles de détection**: Utilisation de techniques d'apprentissage automatique et profond, incluant SVM, Random Forest, LSTM et modèles basés sur les transformateurs comme CamemBERT.
- **Développement d'une application**: Une application interactive permettant de tester les phrases pour la détection de discours de haine.

## Installation

Pour installer et exécuter ce projet localement, veuillez suivre les instructions ci-dessous :

1. Clonez ce dépôt sur votre machine locale :
   ```bash
   git clone https://github.com/votre-utilisateur/votre-repo.git
   ```
2. Naviguez dans le répertoire du projet :
   ```bash
   cd votre-repo
   ```
3. Créez et activez un environnement virtuel :
   ```bash
   python -m venv env
   source env/bin/activate  # Pour Windows: env\Scripts\activate
   ```
4. Installez les dépendances requises :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

1. Prétraitez les données :
   ```bash
   jupyter notebook clean_dataset.ipynb
   ```
2. Entraînez les modèles :
   ```bash
   jupyter notebook training-ML.ipynb
   jupyter notebook training-DL.ipynb
   jupyter notebook training-transformers.ipynb
   ```
3. Testez les modèles :
   ```bash
   jupyter notebook test.ipynb
   ```
4. Exécutez l'application :
   ```bash
   python app_latest.py
   ```

## Contribution

Les contributions sont les bienvenues ! Si vous souhaitez contribuer à ce projet, veuillez suivre les étapes suivantes :

1. Forkez ce dépôt.
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`).
3. Commitez vos changements (`git commit -m 'Add some AmazingFeature'`).
4. Poussez la branche (`git push origin feature/AmazingFeature`).
5. Ouvrez une Pull Request.

## Auteurs

- **Clément Jantet** - [clement.jantet@ecole.ensicaen.fr](mailto:clement.jantet@ecole.ensicaen.fr)
- **Calliste Ravix** - [calliste.ravix@ecole.ensicaen.fr](mailto:calliste.ravix@ecole.ensicaen.fr)

## Remerciements

Nous tenons à remercier nos tuteurs à l'ENSICAEN et à NTNU, ainsi que toutes les personnes qui nous ont soutenus tout au long de ce projet.

## Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.
