# Maintenance prédictive — AI4I 2020

Application Streamlit pour explorer les données, préparer les variables et comparer des modèles de prédiction des pannes.

## Installation

Dans un terminal ouvert dans ce dossier :

```bash
python -m venv .venv
```

Sous Windows :
```bash
.venv\Scripts\activate
```

Installer les dépendances :
```bash
python -m pip install -r requirements.txt
```

## Lancer l'application

```bash
streamlit run app.py
```

Dans l'application, importer le fichier CSV AI4I 2020 via le bouton de téléversement.

## Dataset

L'application demande le CSV via son interface : aucun chemin Kaggle n'est nécessaire dans `app.py`.
