# test.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

print("✅ NumPy version:", np.__version__)

# Test rapide
documents = ["le chat dort", "le chien aboie"]
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(documents)

print("✅ Sklearn fonctionne!")
print(f"✅ Matrice TF-IDF shape: {tfidf.shape}")

print("\n🎉 Tout est prêt ! Tu peux commencer la roadmap.")