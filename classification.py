"""
================================================
classification.py
Sentiment analysis — IMDb Twilight reviews
Labels : positif / moyen / negatif
================================================
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# ================================================
# 1. CHARGEMENT DES DONNÉES
# ================================================
# Corpus nettoyé par nettoyage.py (train 80% / test 20%)

train = pd.read_csv("data/train_clean.csv")
test  = pd.read_csv("data/test_clean.csv")

X_train = train["reviewText_clean"]
y_train = train["label"]
X_test  = test["reviewText_clean"]
y_test  = test["label"]

# ================================================
# 2. VECTORISATION TF-IDF
# ================================================
# TF-IDF : pondère chaque mot selon sa fréquence dans le document
# et sa rareté dans tout le corpus (mots rares = plus informatifs)
#
# ngram_range=(1,2) : prend en compte les unigrammes ET les bigrammes
#   ex: "not good" traité comme un seul token → meilleure gestion de la négation
# min_df=2 : ignore les mots qui apparaissent moins de 2 fois (bruit)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2
    )
X_train_tfidf = vectorizer.fit_transform(X_train) 
X_test_tfidf = vectorizer.transform(X_test)

print(f"Dimensions du corpus vectorisé : {X_train_tfidf.shape}")

# ================================================
# 3. ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES
# ================================================

# ------------------------------------------------
# 3a. Naive Bayes (Multinomial)
# ------------------------------------------------
# Modèle probabiliste basé sur le théorème de Bayes.

nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred_nb = nb.predict(X_test_tfidf)

print("\n=== Naive Bayes ===")
print(classification_report(y_test, y_pred_nb))

# ------------------------------------------------
# 3b. J48 (Decision Tree)
# ------------------------------------------------
# Modèle linéaire qui prédit la probabilité d'appartenance à chaque classe.
# class_weight='balanced' : compense le déséquilibre entre les classes

dt = DecisionTreeClassifier(class_weight='balanced')
dt.fit(X_train_tfidf, y_train)
y_pred_dt = dt.predict(X_test_tfidf)

print("=== Decision Tree (J48) ===")
print(classification_report(y_test, y_pred_dt))

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

feature_names = vectorizer.get_feature_names_out().tolist() 
print(export_text(dt, feature_names=feature_names, max_depth=3))

plt.figure(figsize=(30, 15))
plot_tree(dt, 
          feature_names=feature_names,
          class_names=["moyen", "negatif", "positif"],
          filled=True,
          max_depth=2,  
          fontsize=12)
plt.savefig("arbre_decision.png", dpi=150, bbox_inches='tight')
plt.show()

# ------------------------------------------------
# 3c. Régression Logistique
# ------------------------------------------------
# Modèle linéaire qui prédit la probabilité d'appartenance à chaque classe.
# class_weight='balanced' : compense le déséquilibre entre les classes.
# max_iter=1000 : augmente le nombre d'itérations pour assurer la convergence.

lr = LogisticRegression(max_iter=1000, class_weight='balanced')
lr.fit(X_train_tfidf, y_train)
y_pred_lr = lr.predict(X_test_tfidf)

print("\n=== Régression Logistique ===")
print(classification_report(y_test, y_pred_lr))

# ------------------------------------------------
# 3d. SVM Linéaire
# ------------------------------------------------
# Trouve la frontière de décision qui maximise la marge entre les classes.

svm = LinearSVC(class_weight='balanced')
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)

print("\n=== SVM Linéaire ===")
print(classification_report(y_test, y_pred_svm))