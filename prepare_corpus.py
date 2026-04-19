"""
Ce script sert à la préparation du corpus de reviews IMDb de la masterpiece saga Twilight

Usage:
    python3 prepare_corpus.py

Sorties:
    - corpus.csv        : toutes rétirés du site avec les reviews nettoyées et étiquetées
    - train.csv         : ensemble d'entraînement (80%)
    - test.csv          : ensemble de test (20%)
    - stats_corpus.txt  : résumé du corpus
"""

import pandas as pd
from sklearn.model_selection import train_test_split

FICHIERS = [
    "1ere_serie.csv",
    "2eme_serie.csv",
    "3eme_serie.csv",
    "4eme_serie.csv",
    "5eme_serie.csv",
]

# Classification
# Négatif : 1–3 | Moyen : 4–7 | Positif : 8–10
def assigner_label(note):
    if 1 <= note <= 3:
        return "negatif"
    elif 4 <= note <= 7:
        return "moyen"
    elif 8 <= note <= 10:
        return "positif"
    return None

RANDOM_STATE = 42  
TEST_SIZE    = 0.2

# Chargement et fusion des fichiers
dfs = []
for i, fichier in enumerate(FICHIERS, start=1):
    df = pd.read_csv(fichier, encoding="utf-8")
    df["film"] = f"film_{i}"
    dfs.append(df)
    print(f"  {fichier} : {len(df)} reviews")

corpus = pd.concat(dfs, ignore_index=True)
print(f"\nTotal : {len(corpus)} reviews")

# Pre-nettoyage des reviews pour exclure les reviews sans note
avant = len(corpus)
corpus = corpus.dropna(subset=["rating", "reviewText"])
print(f"Reviews sans note ou sans texte exclues : {avant - len(corpus)}")

# Exclure les textes vides
corpus = corpus[corpus["reviewText"].str.strip() != ""]

# Convertir la note en entier
corpus["rating"] = corpus["rating"].astype(int)

# Attribution des labels
corpus["label"] = corpus["rating"].apply(assigner_label)

# Vérification (ne devrait pas arriver avec des notes 1–10)
corpus = corpus.dropna(subset=["label"])

# Sélection des colonnes utiles
corpus = corpus[["reviewId", "film", "rating", "label", "reviewText"]].reset_index(drop=True)

# Statistiques du corpus complet
print("\nDistribution des labels")
print(corpus["label"].value_counts())
print("\n--- Distribution par film et label ---")
print(corpus.groupby(["film", "label"]).size().unstack(fill_value=0))

# Stratifié : on conserve la même proportion de classes dans train et test
train, test = train_test_split(
    corpus,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=corpus["label"]
)

print(f"\nSplit train/test")
print(f"Train : {len(train)} reviews")
print(f"  {train['label'].value_counts().to_dict()}")
print(f"Test  : {len(test)} reviews")
print(f"  {test['label'].value_counts().to_dict()}")

corpus.to_csv("corpus.csv", index=False, encoding="utf-8")
train.to_csv("train.csv",   index=False, encoding="utf-8")
test.to_csv("test.csv",     index=False, encoding="utf-8")

# Résumé pour faciliter à comprendre notre corpus
with open("résumé_corpus.txt", "w", encoding="utf-8") as f:
    f.write("Corpus IMDb Twilight\n\n")
    f.write(f"Total reviews (après nettoyage) : {len(corpus)}\n")
    f.write(f"Train : {len(train)} | Test : {len(test)}\n\n")
    f.write("Distribution globale des labels :\n")
    f.write(corpus["label"].value_counts().to_string())
    f.write("\n\nDistribution par film :\n")
    f.write(corpus.groupby(["film", "label"]).size().unstack(fill_value=0).to_string())

print("\nFichiers générés : corpus.csv, train.csv, test.csv, résumé_corpus.txt")
print("Prêt pour la nettoyage ! (Utilisez le script nettoyage.py)")
