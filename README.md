# Analyse de sentiment — Saga Twilight (IMDb)

Projet de classification automatique de commentaires IMDb en trois polarités :
**positif**, **moyen**, **négatif**.

## Prérequis

pip install pandas scikit-learn spacy
python -m spacy download en_core_web_sm

## Structure du projet

Fouille-de-texte/
├── data/
│   ├── corpus.csv           # Corpus complet nettoyé et étiqueté
│   ├── train.csv            # Données d'entraînement (80%)
│   ├── test.csv             # Données de test (20%)
│   ├── train_clean.csv      # Données d'entraînement après nettoyage
│   ├── test_clean.csv       # Données de test après nettoyage
│   └── resume_corpus.txt    
├── prepare_corpus.py        # Fusion, étiquetage, split train/test
├── nettoyage.py             # Nettoyage et lemmatisation du texte
├── classification.py        # Vectorisation TF-IDF + classification
└── README.md

## Exécution

# Étape 1 — Préparer le corpus
python prepare_corpus.py

# Étape 2 — Nettoyer le texte
python nettoyage.py

# Étape 3 — Vectorisation et classification
python classification.py