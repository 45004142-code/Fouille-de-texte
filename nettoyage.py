"""
Usage:
    python3 nettoyage.py

Entrées:
    - train.csv
    - test.csv

Sorties:
    - train_clean.csv
    - test_clean.csv
"""

import re
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")
nlp.disable_pipes(["ner", "parser"]) # On désactive les composants inutiles pour que ce soit plus vite

def nettoyer_texte(texte):
    if not isinstance(texte, str) or texte.strip() == "":
        return ""

    # Suppression des balises HTML
    texte = re.sub(r"<[^>]+>", " ", texte)

    # Suppression des caractères non alphabétiques (chiffres, ponctuation etc)
    texte = re.sub(r"[^a-zA-Z\s]", " ", texte)

    # Mettre le texte en minuscules
    texte = texte.lower()

    # Suppression des espaces multiples
    texte = re.sub(r"\s+", " ", texte).strip()

    # lemmatisation + stopwords avec SpaCy
    doc = nlp(texte)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop       
        and not token.is_punct     
        and len(token.lemma_) > 1
    ]

    return " ".join(tokens)

# Chargement des données
train = pd.read_csv("train.csv", encoding="utf-8")
test  = pd.read_csv("test.csv",  encoding="utf-8")

print(f"  Train : {len(train)} reviews")
print(f"  Test  : {len(test)} reviews")

# Nettoyage

print("\nEn train de faire la nettoyage (cela peut prendre quelques minutes)")
train["reviewText_clean"] = train["reviewText"].apply(nettoyer_texte)
test["reviewText_clean"] = test["reviewText"].apply(nettoyer_texte)

# On vérifie : suppression des reviews vides après nettoyage

avant_train = len(train)
avant_test  = len(test)

train = train[train["reviewText_clean"].str.strip() != ""].reset_index(drop=True)
test  = test[test["reviewText_clean"].str.strip()  != ""].reset_index(drop=True)

if avant_train - len(train) > 0:
    print(f"  Reviews vides supprimées du train : {avant_train - len(train)}")
if avant_test - len(test) > 0:
    print(f"  Reviews vides supprimées du test  : {avant_test - len(test)}")

# Aperçu du résultat pour savoir si ça a bien marché
print("\n--- Aperçu avant/après nettoyage ---")
for _, row in train.head(3).iterrows():
    print(f"\nOriginal  : {row['reviewText'][:150]}...")
    print(f"Nettoyé   : {row['reviewText_clean'][:150]}...")

# On sauvegarde le corpus nettoyé
train.to_csv("train_clean.csv", index=False, encoding="utf-8")
test.to_csv("test_clean.csv",   index=False, encoding="utf-8")

print("\nFichiers générés : train_clean.csv, test_clean.csv")
print("Prêt pour la vectorisation !")
