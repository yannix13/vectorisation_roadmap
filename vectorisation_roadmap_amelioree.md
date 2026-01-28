# 🗺️ ROADMAP DÉTAILLÉE : PHASE 1 - Représenter du texte

**Objectif final :** Comprendre comment transformer du texte en vecteurs et construire un moteur de recherche qui s'améliore progressivement.

**Durée totale :** 3-4 semaines (flexible selon ton rythme)

---

## 📅 SEMAINE 1 : Bag of Words (BoW)

### 🎯 Objectif de la semaine
Comprendre la méthode la plus simple pour transformer texte → vecteur

---

### JOUR 1 : Théorie + Compréhension conceptuelle

#### 📖 Concepts à maîtriser

**1. Qu'est-ce que Bag of Words ?**

Idée : Représenter un texte par le compte de ses mots, en ignorant l'ordre.

**Exemple concret :**

```
Vocabulaire : ["chat", "chien", "mignon", "dort", "aboie"]

Document 1 : "le chat dort"
→ Vecteur : [1, 0, 0, 1, 0]
           chat chien mignon dort aboie

Document 2 : "le chien aboie"
→ Vecteur : [0, 1, 0, 0, 1]

Document 3 : "le chat mignon dort"
→ Vecteur : [1, 0, 1, 1, 0]
```

**2. Les étapes de BoW :**

```
Étape 1 : Créer le vocabulaire
          Lister tous les mots uniques de tous les documents

Étape 2 : Compter les occurrences
          Pour chaque document, compter combien de fois 
          chaque mot du vocabulaire apparaît

Étape 3 : Créer le vecteur
          Chaque position = un mot du vocabulaire
          Chaque valeur = nombre d'occurrences
```

**3. Similarité entre documents**

Utiliser la similarité cosinus (que tu connais déjà !)

```python
# Rappel de la formule
similarité = (doc1 · doc2) / (||doc1|| × ||doc2||)
```

---

#### ✏️ Exercices papier (30-45 min)

**Exercice 1 : Créer un vocabulaire**

Documents :
- "le chat dort"
- "le chien court"
- "le chat court"

→ Quel est le vocabulaire ? (ignore "le")

<details>
<summary>Solution</summary>

Vocabulaire : ["chat", "chien", "dort", "court"]

</details>

---

**Exercice 2 : Vectorisation manuelle**

Avec le vocabulaire ci-dessus, transforme en vecteurs :
- "le chat dort"
- "le chien court"
- "le chat court"

<details>
<summary>Solution</summary>

- `[1, 0, 1, 0]`
- `[0, 1, 0, 1]`
- `[1, 0, 0, 1]`

</details>

---

**Exercice 3 : Calcul de similarité**

Quelle paire de documents est la plus similaire ? Calcule manuellement la similarité cosinus.

---

### JOUR 2-3 : Implémentation de base (à la main)

#### 💻 Projet : Coder BoW sans librairie

**Objectif :** Comprendre chaque étape en la codant toi-même

```python
# bow_from_scratch.py

def create_vocabulary(documents):
    """
    Crée le vocabulaire à partir d'une liste de documents
    
    Args:
        documents: liste de strings ["doc1", "doc2", ...]
    
    Returns:
        liste de mots uniques, triée alphabétiquement
    """
    # TODO : 
    # 1. Mettre tous les mots en minuscules
    # 2. Séparer les mots (split)
    # 3. Créer un ensemble de mots uniques
    # 4. Trier et retourner comme liste
    pass


def document_to_vector(document, vocabulary):
    """
    Transforme un document en vecteur BoW
    
    Args:
        document: string
        vocabulary: liste de mots
    
    Returns:
        liste de nombres (vecteur)
    """
    # TODO :
    # 1. Initialiser un vecteur de zéros (longueur = taille vocabulaire)
    # 2. Pour chaque mot du document :
    #    - Trouver son index dans vocabulary
    #    - Incrémenter la valeur à cet index
    # 3. Retourner le vecteur
    pass


def cosine_similarity(vec1, vec2):
    """
    Calcule la similarité cosinus entre deux vecteurs
    
    Args:
        vec1, vec2: listes de nombres
    
    Returns:
        float entre 0 et 1
    """
    # TODO :
    # 1. Calculer le produit scalaire (dot product)
    # 2. Calculer les normes
    # 3. Diviser : dot / (norm1 * norm2)
    pass


# Test
if __name__ == "__main__":
    documents = [
        "le chat dort sur le tapis",
        "le chien aboie dans le jardin",
        "le chat mignon joue avec la souris"
    ]
    
    # Créer vocabulaire
    vocab = create_vocabulary(documents)
    print("Vocabulaire:", vocab)
    
    # Vectoriser
    vectors = [document_to_vector(doc, vocab) for doc in documents]
    print("\nVecteurs:")
    for i, vec in enumerate(vectors):
        print(f"Doc {i+1}: {vec}")
    
    # Tester similarité
    sim = cosine_similarity(vectors[0], vectors[2])
    print(f"\nSimilarité doc1 vs doc3: {sim:.3f}")
```

---

#### 📝 Étapes détaillées

**Étape 1 : create_vocabulary() (30-45 min)**

Indices :

```python
# Astuce 1 : mettre en minuscules
text.lower()

# Astuce 2 : séparer en mots
text.split()

# Astuce 3 : ensemble de mots uniques
set(words)

# Astuce 4 : trier
sorted(word_set)
```

---

**Étape 2 : document_to_vector() (45-60 min)**

Indices :

```python
# Initialiser vecteur de zéros
vector = [0] * len(vocabulary)

# Trouver index d'un mot
index = vocabulary.index(word)

# Incrémenter
vector[index] += 1
```

---

**Étape 3 : cosine_similarity() (30 min)**

Indices :

```python
import math

# Produit scalaire
dot = sum(a * b for a, b in zip(vec1, vec2))

# Norme
norm = math.sqrt(sum(x**2 for x in vec))
```

---

#### ✅ Checkpoint Jour 2-3

Tu dois avoir :

- ✓ Un fichier `bow_from_scratch.py` qui fonctionne
- ✓ Testé avec au moins 3 documents
- ✓ Compris chaque ligne de code que tu as écrite
- ✓ Vérifié que la similarité donne des résultats cohérents

---

### JOUR 4-5 : Mini moteur de recherche v0.1

#### 💻 Projet : Premier moteur de recherche

**Objectif :** Créer un outil utilisable

```python
# search_engine_v01.py

class SimpleSearchEngine:
    def __init__(self, documents):
        """
        Initialise le moteur avec une collection de documents
        
        Args:
            documents: liste de strings
        """
        self.documents = documents
        self.vocabulary = create_vocabulary(documents)
        self.doc_vectors = [
            document_to_vector(doc, self.vocabulary) 
            for doc in documents
        ]
    
    def search(self, query, top_k=3):
        """
        Recherche les documents les plus pertinents
        
        Args:
            query: string (requête de l'utilisateur)
            top_k: nombre de résultats à retourner
        
        Returns:
            liste de tuples (index_document, score_similarité)
        """
        # TODO :
        # 1. Transformer la requête en vecteur
        # 2. Calculer similarité avec chaque document
        # 3. Trier par score décroissant
        # 4. Retourner top_k résultats
        pass
    
    def display_results(self, results):
        """
        Affiche les résultats de manière lisible
        """
        print("\n" + "="*50)
        print("RÉSULTATS DE RECHERCHE")
        print("="*50)
        for rank, (doc_idx, score) in enumerate(results, 1):
            print(f"\n{rank}. Score: {score:.3f}")
            print(f"   {self.documents[doc_idx][:100]}...")
        print("="*50)


# Test interactif
if __name__ == "__main__":
    # Collection de documents
    documents = [
        "Le chat dort paisiblement sur le canapé",
        "Le chien aboie fort dans le jardin",
        "Le chat mignon joue avec une pelote de laine",
        "Les chiens courent dans le parc",
        "Le chaton mange ses croquettes",
        "Python est un langage de programmation",
        "Java est utilisé pour développer des applications",
        "Le machine learning utilise des algorithmes"
    ]
    
    # Créer le moteur
    engine = SimpleSearchEngine(documents)
    
    # Boucle de recherche
    while True:
        query = input("\nRecherche (ou 'quit' pour quitter): ")
        if query.lower() == 'quit':
            break
        
        results = engine.search(query, top_k=3)
        engine.display_results(results)
```

---

#### 📝 Étapes détaillées

**Étape 1 : Implémenter search() (1-2h)**

```python
def search(self, query, top_k=3):
    # 1. Vectoriser la requête
    query_vector = document_to_vector(query, self.vocabulary)
    
    # 2. Calculer toutes les similarités
    similarities = []
    for i, doc_vec in enumerate(self.doc_vectors):
        sim = cosine_similarity(query_vector, doc_vec)
        similarities.append((i, sim))
    
    # 3. Trier par score décroissant
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 4. Retourner top_k
    return similarities[:top_k]
```

---

**Étape 2 : Tests et observations (30 min)**

Teste avec :
- "chat" → devrait trouver les docs avec des chats
- "programmation" → devrait trouver les docs techniques
- "chaton mignon" → que se passe-t-il ?

---

#### ✅ Checkpoint Jour 4-5

Tu dois avoir :

- ✓ Un moteur de recherche fonctionnel
- ✓ Testé avec 10+ requêtes différentes
- ✓ Noté les forces et faiblesses de BoW
- ✓ Liste de ce qui pourrait être amélioré

---

### JOUR 6-7 : Optimisation et expérimentation

#### 🔬 Améliorations à tester

**1. Prétraitement du texte**

```python
import re
from collections import Counter

def preprocess_text(text):
    """
    Nettoie le texte avant vectorisation
    """
    # Minuscules
    text = text.lower()
    
    # Retirer la ponctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Séparer les mots
    words = text.split()
    
    # Filtrer les stop words (mots vides)
    stop_words = {'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 
                  'et', 'ou', 'dans', 'sur', 'avec', 'pour'}
    words = [w for w in words if w not in stop_words]
    
    return ' '.join(words)
```

---

**2. Stemming / Lemmatisation**

```python
# Simple stemmer (enlève les terminaisons)
def simple_stem(word):
    """
    Version très simplifiée de stemming
    """
    suffixes = ['tion', 'ment', 'eur', 'euse', 's', 'x']
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

# Exemple d'utilisation
words = ["programmation", "programmer", "programmes"]
stems = [simple_stem(w) for w in words]
# Tous deviennent "programm"
```

---

**3. Métriques d'évaluation**

```python
def evaluate_search_engine(engine, test_queries):
    """
    Évalue la qualité du moteur de recherche
    
    Args:
        engine: moteur de recherche
        test_queries: liste de (query, expected_doc_indices)
    
    Returns:
        précision moyenne
    """
    precisions = []
    
    for query, expected_docs in test_queries:
        results = engine.search(query, top_k=3)
        result_indices = [idx for idx, score in results]
        
        # Calculer précision
        correct = len(set(result_indices) & set(expected_docs))
        precision = correct / len(result_indices)
        precisions.append(precision)
    
    return sum(precisions) / len(precisions)


# Exemple d'utilisation
test_queries = [
    ("chat", [0, 2, 4]),  # docs avec des chats
    ("programmation", [5, 6, 7]),  # docs techniques
]

precision = evaluate_search_engine(engine, test_queries)
print(f"Précision moyenne: {precision:.2%}")
```

---

#### 🎨 Projets bonus (optionnels)

**Projet 1 : Visualisation**

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_documents(doc_vectors, documents):
    """
    Visualise les documents en 2D
    """
    # Réduire à 2 dimensions
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(doc_vectors)
    
    # Créer le graphique
    plt.figure(figsize=(10, 8))
    plt.scatter(coords_2d[:, 0], coords_2d[:, 1])
    
    # Annoter chaque point
    for i, (x, y) in enumerate(coords_2d):
        plt.annotate(f"Doc{i}", (x, y), fontsize=8)
    
    plt.title("Documents dans l'espace vectoriel (2D)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.show()
```

---

**Projet 2 : Interface utilisateur simple**

```python
def interactive_search():
    """
    Interface en ligne de commande améliorée
    """
    print("="*60)
    print("MOTEUR DE RECHERCHE v0.1")
    print("="*60)
    print("\nCommandes disponibles:")
    print("  - search <requête> : rechercher")
    print("  - stats : statistiques du corpus")
    print("  - quit : quitter")
    print("="*60)
    
    while True:
        cmd = input("\n> ").strip()
        
        if cmd.startswith("search "):
            query = cmd[7:]
            results = engine.search(query, top_k=3)
            engine.display_results(results)
        
        elif cmd == "stats":
            print(f"\nCorpus: {len(engine.documents)} documents")
            print(f"Vocabulaire: {len(engine.vocabulary)} mots")
        
        elif cmd == "quit":
            print("Au revoir !")
            break
        
        else:
            print("Commande inconnue")
```

---

#### ✅ Checkpoint Semaine 1

**Compétences acquises :**

- ✓ Comprendre Bag of Words conceptuellement
- ✓ Implémenter BoW from scratch
- ✓ Construire un moteur de recherche basique
- ✓ Identifier les limites de BoW
- ✓ Optimiser le prétraitement

**Questions de réflexion :**

1. Pourquoi BoW perd-il l'ordre des mots ?
2. Quels types de requêtes fonctionnent mal avec BoW ?
3. Comment pourrait-on améliorer BoW ?

---

## 📅 SEMAINE 2 : TF-IDF

### 🎯 Objectif de la semaine

Comprendre comment valoriser les mots importants et réduire l'impact des mots courants

---

### JOUR 1 : Comprendre TF-IDF

#### 📖 Concepts à maîtriser

**1. Pourquoi TF-IDF ?**

**Problème avec BoW :**
```
Document : "le chat dort sur le canapé"
Vecteur BoW : [2, 1, 1, 1, 1, 1]  ← "le" compte pour 2 !
              le chat dort sur canapé
```

"le" apparaît souvent mais n'apporte pas d'information.

**Solution :** Pondérer les mots selon leur importance !

---

**2. Les deux composantes de TF-IDF**

**TF (Term Frequency) :**
```
TF(mot, document) = nombre d'occurrences du mot dans le document
                    ───────────────────────────────────────────
                    nombre total de mots dans le document
```

**IDF (Inverse Document Frequency) :**
```
IDF(mot, corpus) = log( nombre total de documents          )
                       ─────────────────────────────────────
                       nombre de documents contenant le mot
```

**TF-IDF final :**
```
TF-IDF(mot, doc, corpus) = TF(mot, doc) × IDF(mot, corpus)
```

---

**3. Intuition avec un exemple**

Corpus :
- Doc1 : "le chat dort"
- Doc2 : "le chien court"
- Doc3 : "le chat mignon"

Pour le mot "le" :
```
TF(le, Doc1) = 1/3 = 0.33
IDF(le, corpus) = log(3/3) = log(1) = 0
TF-IDF(le, Doc1) = 0.33 × 0 = 0  ← "le" est neutralisé !
```

Pour le mot "chat" :
```
TF(chat, Doc1) = 1/3 = 0.33
IDF(chat, corpus) = log(3/2) = 0.176
TF-IDF(chat, Doc1) = 0.33 × 0.176 = 0.058  ← score > 0
```

---

#### ✏️ Exercices papier (45 min)

**Exercice 1 : Calcul manuel de TF**

Document : "le chat dort sur le canapé"

Calcule TF pour :
- "le"
- "chat"
- "dort"

<details>
<summary>Solution</summary>

Nombre total de mots = 6

- TF(le) = 2/6 = 0.33
- TF(chat) = 1/6 = 0.17
- TF(dort) = 1/6 = 0.17

</details>

---

**Exercice 2 : Calcul manuel d'IDF**

Corpus :
- Doc1 : "chat dort"
- Doc2 : "chat joue"
- Doc3 : "chien court"
- Doc4 : "chien aboie"

Calcule IDF pour :
- "chat" (présent dans 2 docs)
- "chien" (présent dans 2 docs)
- "dort" (présent dans 1 doc)

<details>
<summary>Solution</summary>

Nombre total de documents = 4

- IDF(chat) = log(4/2) = log(2) = 0.693
- IDF(chien) = log(4/2) = log(2) = 0.693
- IDF(dort) = log(4/1) = log(4) = 1.386

</details>

---

**Exercice 3 : TF-IDF complet**

Avec les valeurs ci-dessus, calcule TF-IDF(chat, Doc1).

Sachant que Doc1 = "chat dort" (2 mots).

<details>
<summary>Solution</summary>

```
TF(chat, Doc1) = 1/2 = 0.5
IDF(chat) = 0.693
TF-IDF(chat, Doc1) = 0.5 × 0.693 = 0.347
```

</details>

---

### JOUR 2-3 : Implémentation de TF-IDF

#### 💻 Projet : Coder TF-IDF from scratch

```python
# tfidf_from_scratch.py

import math
from collections import Counter

def compute_tf(document):
    """
    Calcule le Term Frequency pour chaque mot du document
    
    Args:
        document: string
    
    Returns:
        dict {mot: tf_score}
    """
    words = document.lower().split()
    word_count = Counter(words)
    total_words = len(words)
    
    tf = {}
    for word, count in word_count.items():
        tf[word] = count / total_words
    
    return tf


def compute_idf(documents):
    """
    Calcule l'IDF pour chaque mot du corpus
    
    Args:
        documents: liste de strings
    
    Returns:
        dict {mot: idf_score}
    """
    num_documents = len(documents)
    
    # Compter dans combien de documents chaque mot apparaît
    word_doc_count = Counter()
    for doc in documents:
        words = set(doc.lower().split())
        for word in words:
            word_doc_count[word] += 1
    
    # Calculer IDF
    idf = {}
    for word, doc_count in word_doc_count.items():
        idf[word] = math.log(num_documents / doc_count)
    
    return idf


def compute_tfidf(document, idf_scores):
    """
    Calcule le vecteur TF-IDF d'un document
    
    Args:
        document: string
        idf_scores: dict {mot: idf}
    
    Returns:
        dict {mot: tfidf_score}
    """
    tf = compute_tf(document)
    
    tfidf = {}
    for word, tf_score in tf.items():
        if word in idf_scores:
            tfidf[word] = tf_score * idf_scores[word]
    
    return tfidf


def documents_to_tfidf_vectors(documents, vocabulary):
    """
    Transforme tous les documents en vecteurs TF-IDF
    
    Args:
        documents: liste de strings
        vocabulary: liste de mots (ordre des dimensions)
    
    Returns:
        liste de listes (vecteurs TF-IDF)
    """
    # Calculer IDF une seule fois pour tout le corpus
    idf_scores = compute_idf(documents)
    
    # Vectoriser chaque document
    vectors = []
    for doc in documents:
        tfidf = compute_tfidf(doc, idf_scores)
        
        # Créer vecteur dans l'ordre du vocabulaire
        vector = [tfidf.get(word, 0.0) for word in vocabulary]
        vectors.append(vector)
    
    return vectors


# Test
if __name__ == "__main__":
    documents = [
        "le chat dort sur le tapis",
        "le chien aboie dans le jardin",
        "le chat mignon joue avec la souris"
    ]
    
    # Créer vocabulaire (sans stop words cette fois)
    all_words = []
    for doc in documents:
        all_words.extend(doc.lower().split())
    
    stop_words = {'le', 'la', 'les', 'dans', 'sur', 'avec'}
    vocabulary = sorted(set(w for w in all_words if w not in stop_words))
    
    print("Vocabulaire:", vocabulary)
    
    # Calculer IDF
    idf = compute_idf(documents)
    print("\nIDF scores:")
    for word in vocabulary:
        print(f"  {word}: {idf.get(word, 0):.3f}")
    
    # Vectoriser
    tfidf_vectors = documents_to_tfidf_vectors(documents, vocabulary)
    print("\nVecteurs TF-IDF:")
    for i, vec in enumerate(tfidf_vectors):
        print(f"\nDoc {i+1}:")
        for word, score in zip(vocabulary, vec):
            if score > 0:
                print(f"  {word}: {score:.3f}")
```

---

#### 📝 Points d'attention

**1. Gestion des mots absents**

```python
# Si un mot de la requête n'est pas dans le corpus
if word in idf_scores:
    tfidf[word] = tf[word] * idf_scores[word]
else:
    tfidf[word] = 0  # ou ignorer le mot
```

---

**2. Normalisation des vecteurs**

```python
def normalize_vector(vector):
    """
    Normalise un vecteur (norme = 1)
    """
    norm = math.sqrt(sum(x**2 for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]
```

---

**3. Variantes de TF-IDF**

```python
# TF logarithmique
tf_log = 1 + math.log(count) if count > 0 else 0

# IDF smooth
idf_smooth = math.log((1 + num_documents) / (1 + doc_count)) + 1

# TF-IDF normalisé
tfidf_normalized = normalize_vector(tfidf_vector)
```

---

#### ✅ Checkpoint Jour 2-3

Tu dois avoir :

- ✓ Implémentation complète de TF-IDF
- ✓ Tests sur plusieurs documents
- ✓ Comparaison des scores TF vs TF-IDF
- ✓ Vérification que les mots courants ont des scores faibles

---

### JOUR 4-5 : Moteur de recherche v0.2

#### 💻 Projet : Améliorer le moteur avec TF-IDF

```python
# search_engine_v02.py

class TfidfSearchEngine:
    def __init__(self, documents):
        """
        Initialise le moteur TF-IDF
        """
        self.documents = documents
        
        # Créer vocabulaire (sans stop words)
        stop_words = {'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du',
                      'et', 'ou', 'dans', 'sur', 'avec', 'pour', 'par'}
        
        all_words = []
        for doc in documents:
            words = doc.lower().split()
            all_words.extend(w for w in words if w not in stop_words)
        
        self.vocabulary = sorted(set(all_words))
        
        # Calculer IDF
        self.idf_scores = compute_idf(documents)
        
        # Vectoriser tous les documents
        self.doc_vectors = documents_to_tfidf_vectors(
            documents, 
            self.vocabulary
        )
        
        # Normaliser les vecteurs
        self.doc_vectors = [
            normalize_vector(vec) for vec in self.doc_vectors
        ]
    
    def search(self, query, top_k=3):
        """
        Recherche avec TF-IDF
        """
        # Vectoriser la requête
        query_tfidf = compute_tfidf(query, self.idf_scores)
        query_vector = [query_tfidf.get(word, 0.0) for word in self.vocabulary]
        query_vector = normalize_vector(query_vector)
        
        # Calculer similarités
        similarities = []
        for i, doc_vec in enumerate(self.doc_vectors):
            sim = cosine_similarity(query_vector, doc_vec)
            similarities.append((i, sim))
        
        # Trier et retourner
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def compare_with_bow(self, query):
        """
        Compare les résultats TF-IDF vs BoW
        """
        print("\n" + "="*70)
        print(f"COMPARAISON : '{query}'")
        print("="*70)
        
        # Résultats TF-IDF
        tfidf_results = self.search(query, top_k=3)
        
        # Résultats BoW (simulation simple)
        bow_engine = SimpleSearchEngine(self.documents)
        bow_results = bow_engine.search(query, top_k=3)
        
        print("\nTF-IDF:")
        for rank, (idx, score) in enumerate(tfidf_results, 1):
            print(f"  {rank}. [{score:.3f}] {self.documents[idx][:60]}...")
        
        print("\nBoW:")
        for rank, (idx, score) in enumerate(bow_results, 1):
            print(f"  {rank}. [{score:.3f}] {self.documents[idx][:60]}...")
        
        print("="*70)


# Test
if __name__ == "__main__":
    documents = [
        "Le chat dort paisiblement sur le canapé moelleux",
        "Le chien aboie fort dans le jardin ensoleillé",
        "Le chat mignon joue avec une pelote de laine rouge",
        "Les chiens courent rapidement dans le grand parc",
        "Le petit chaton mange ses croquettes avec appétit",
        "Python est un langage de programmation populaire",
        "Java est utilisé pour développer des applications robustes",
        "Le machine learning utilise des algorithmes complexes",
        "Les réseaux de neurones sont puissants pour l'IA",
        "Le deep learning révolutionne l'intelligence artificielle"
    ]
    
    # Créer moteur TF-IDF
    engine = TfidfSearchEngine(documents)
    
    # Tests de comparaison
    test_queries = [
        "chat mignon",
        "programmation Python",
        "chien dans jardin",
        "intelligence artificielle"
    ]
    
    for query in test_queries:
        engine.compare_with_bow(query)
        input("\nAppuyez sur Entrée pour continuer...")
```

---

#### 📊 Analyse des résultats

**Ce que tu dois observer :**

1. **Mots rares valorisés :**
   - Requête : "programmation Python"
   - TF-IDF privilégie les docs techniques
   - BoW peut être noyé par "le", "la", etc.

2. **Meilleure discrimination :**
   - TF-IDF distingue mieux les sujets
   - Scores plus variés

3. **Limites qui restent :**
   - Ordre des mots toujours perdu
   - Synonymes non gérés
   - "chat" ≠ "chaton"

---

#### ✅ Checkpoint Jour 4-5

Tu dois avoir :

- ✓ Moteur TF-IDF fonctionnel
- ✓ Comparaison avec BoW
- ✓ Compris pourquoi TF-IDF est meilleur
- ✓ Identifié ce qui reste à améliorer

---

### JOUR 6-7 : sklearn et optimisations

#### 💻 Utiliser sklearn (la bonne façon)

**Pourquoi passer à sklearn ?**

- Code optimisé (10-100x plus rapide)
- Fonctionnalités avancées
- Standard de l'industrie

```python
# tfidf_with_sklearn.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TfidfSearchEngineSKLearn:
    def __init__(self, documents):
        """
        Moteur TF-IDF avec sklearn
        """
        self.documents = documents
        
        # Créer le vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,          # minuscules
            stop_words='french',     # stop words français
            max_df=0.8,             # ignorer mots trop fréquents (>80%)
            min_df=2,               # ignorer mots trop rares (<2 docs)
            ngram_range=(1, 2)      # unigrammes et bigrammes
        )
        
        # Vectoriser le corpus
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        print(f"Vocabulaire: {len(self.vectorizer.vocabulary_)} mots")
        print(f"Matrice: {self.tfidf_matrix.shape}")
    
    def search(self, query, top_k=5):
        """
        Recherche vectorielle
        """
        # Vectoriser la requête
        query_vec = self.vectorizer.transform([query])
        
        # Calculer similarités (optimisé!)
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)
        
        # Obtenir top_k indices
        top_indices = similarities.argsort()[0][-top_k:][::-1]
        
        # Retourner résultats
        results = []
        for idx in top_indices:
            score = similarities[0, idx]
            results.append((idx, score))
        
        return results
    
    def get_top_terms(self, doc_idx, top_n=10):
        """
        Obtient les mots les plus importants d'un document
        """
        # Vecteur du document
        doc_vec = self.tfidf_matrix[doc_idx]
        
        # Obtenir les scores
        feature_names = self.vectorizer.get_feature_names_out()
        scores = doc_vec.toarray()[0]
        
        # Trier
        top_indices = scores.argsort()[-top_n:][::-1]
        
        top_terms = []
        for idx in top_indices:
            if scores[idx] > 0:
                top_terms.append((feature_names[idx], scores[idx]))
        
        return top_terms
    
    def analyze_query(self, query):
        """
        Analyse une requête (débug)
        """
        query_vec = self.vectorizer.transform([query])
        feature_names = self.vectorizer.get_feature_names_out()
        scores = query_vec.toarray()[0]
        
        print(f"\nAnalyse de la requête: '{query}'")
        print("-" * 50)
        
        query_terms = []
        for idx, score in enumerate(scores):
            if score > 0:
                query_terms.append((feature_names[idx], score))
        
        query_terms.sort(key=lambda x: x[1], reverse=True)
        
        for term, score in query_terms:
            print(f"  {term}: {score:.3f}")


# Test avec un corpus plus grand
if __name__ == "__main__":
    # Corpus étendu
    documents = [
        "Le chat dort paisiblement sur le canapé moelleux",
        "Le chien aboie fort dans le jardin ensoleillé",
        "Le chat mignon joue avec une pelote de laine rouge",
        "Les chiens courent rapidement dans le grand parc",
        "Le petit chaton mange ses croquettes avec appétit",
        "Python est un langage de programmation populaire et facile à apprendre",
        "Java est utilisé pour développer des applications robustes et scalables",
        "Le machine learning utilise des algorithmes complexes et des données",
        "Les réseaux de neurones sont puissants pour l'intelligence artificielle",
        "Le deep learning révolutionne l'intelligence artificielle moderne",
        "Les modèles de langage comme GPT sont impressionnants",
        "La programmation Python est idéale pour le data science",
        "Les algorithmes de tri sont fondamentaux en informatique",
        "Le développement web utilise JavaScript et Python",
        "Les bases de données SQL sont essentielles pour les applications"
    ]
    
    # Créer moteur
    engine = TfidfSearchEngineSKLearn(documents)
    
    # Tests
    queries = [
        "chat mignon",
        "algorithmes machine learning",
        "développement applications",
        "chien parc"
    ]
    
    for query in queries:
        print("\n" + "="*70)
        results = engine.search(query, top_k=3)
        
        # Afficher requête
        engine.analyze_query(query)
        
        # Afficher résultats
        print("\nRésultats:")
        for rank, (idx, score) in enumerate(results, 1):
            print(f"\n{rank}. Score: {score:.3f}")
            print(f"   {documents[idx]}")
            
            # Top termes du document
            top_terms = engine.get_top_terms(idx, top_n=5)
            print(f"   Termes clés: {', '.join(t for t, s in top_terms)}")
        
        print("="*70)
        input("\nAppuyez sur Entrée...")
```

---

#### 🎨 Fonctionnalités avancées

**1. N-grammes**

```python
# Capturer des expressions
# "machine learning" au lieu de "machine" + "learning"

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)  # unigrammes et bigrammes
)

# Génère :
# - "machine"
# - "learning"
# - "machine learning"
```

---

**2. Filtrage par fréquence**

```python
vectorizer = TfidfVectorizer(
    max_df=0.8,   # ignorer mots dans >80% des docs
    min_df=2      # ignorer mots dans <2 docs
)
```

---

**3. Analyseur personnalisé**

```python
import re

def custom_analyzer(text):
    """
    Analyseur personnalisé
    """
    # Minuscules
    text = text.lower()
    
    # Tokenization
    tokens = re.findall(r'\b\w+\b', text)
    
    # Stemming simple
    tokens = [simple_stem(t) for t in tokens]
    
    # Filtrer stop words
    stop_words = {'le', 'la', 'les', ...}
    tokens = [t for t in tokens if t not in stop_words]
    
    return tokens

vectorizer = TfidfVectorizer(
    analyzer=custom_analyzer
)
```

---

#### 📊 Benchmarking

```python
import time

def benchmark_search(engine, queries, iterations=100):
    """
    Mesure la performance
    """
    start = time.time()
    
    for _ in range(iterations):
        for query in queries:
            engine.search(query, top_k=5)
    
    elapsed = time.time() - start
    
    avg_time = elapsed / (iterations * len(queries))
    
    print(f"Temps moyen par requête: {avg_time*1000:.2f} ms")
    print(f"Requêtes par seconde: {1/avg_time:.0f}")


# Test
test_queries = ["chat", "programmation", "intelligence artificielle"]

print("Custom implementation:")
benchmark_search(custom_engine, test_queries)

print("\nsklearn implementation:")
benchmark_search(sklearn_engine, test_queries)
```

---

#### ✅ Checkpoint Semaine 2

**Compétences acquises :**

- ✓ Comprendre TF-IDF en profondeur
- ✓ Implémenter TF-IDF from scratch
- ✓ Utiliser sklearn efficacement
- ✓ Optimiser un moteur de recherche
- ✓ Benchmarker et comparer

**Questions de réflexion :**

1. En quoi TF-IDF est-il supérieur à BoW ?
2. Quelles limites restent malgré TF-IDF ?
3. Comment gérer les synonymes et la polysémie ?

---

## 📅 SEMAINE 3 : Word2Vec

### 🎯 Objectif de la semaine

Comprendre les word embeddings : représentations vectorielles denses qui capturent le sens des mots

---

### JOUR 1 : Théorie des word embeddings

#### 📖 Concepts à maîtriser

**1. Le problème avec TF-IDF**

```
Problèmes :
✗ Vecteurs creux (beaucoup de zéros)
✗ Grande dimensionnalité (taille = taille vocabulaire)
✗ Pas de notion de similarité sémantique
   "chat" et "chaton" sont orthogonaux !
```

**Exemple :**

```python
# Avec TF-IDF
vocab = ["chat", "chaton", "chien", "voiture", ...]  # 10,000 mots

vec_chat   = [0.5, 0,   0,   0,   ...]  # 10,000 dimensions
vec_chaton = [0,   0.5, 0,   0,   ...]  
# → similarité = 0 ! 😢
```

---

**2. L'idée des word embeddings**

**Hypothèse distributionnelle :**

> "Un mot est caractérisé par la compagnie qu'il tient"
> — J.R. Firth

**Traduction :** Des mots qui apparaissent dans des contextes similaires ont des sens similaires.

```
"Le chat dort sur le canapé"
"Le chaton dort sur le lit"
"Le chien dort dans sa niche"

→ "chat", "chaton", "chien" apparaissent dans des contextes similaires
→ Ils devraient avoir des vecteurs similaires !
```

---

**3. Word2Vec : l'idée géniale**

**Objectif :** Apprendre des vecteurs denses (100-300 dimensions) où :
- Des mots similaires ont des vecteurs proches
- Les relations sémantiques sont capturées

```python
# Avec Word2Vec
vec_chat   = [0.2, -0.5, 0.8, 0.1, ..., -0.3]  # 300 dimensions
vec_chaton = [0.3, -0.4, 0.7, 0.2, ..., -0.2]
# → similarité = 0.89 ! 😊
```

---

**4. Les deux architectures Word2Vec**

**a) CBOW (Continuous Bag of Words)**

```
Idée : Prédire le mot central à partir du contexte

Exemple : "le [?] dort sur le"
          → devrait prédire "chat"

Contexte : [le, dort, sur, le]  →  Modèle  →  Prédiction : chat
```

**b) Skip-gram**

```
Idée : Prédire le contexte à partir du mot central

Exemple : Mot = "chat"
          → devrait prédire : [le, dort, sur, le]

Mot : chat  →  Modèle  →  Prédiction : [le, dort, sur, le]
```

**Quelle architecture choisir ?**
- CBOW : plus rapide, mieux pour mots fréquents
- Skip-gram : meilleur pour mots rares (qu'on utilise le plus)

---

#### ✏️ Exercices conceptuels

**Exercice 1 : Identifier le contexte**

Phrase : "Le chat mignon dort sur le canapé moelleux"

Pour le mot "dort" (fenêtre de contexte = 2) :
- Quels sont les mots de contexte ?

<details>
<summary>Solution</summary>

Contexte de "dort" :
- Gauche : "chat", "mignon"
- Droite : "sur", "le"

Contexte complet : ["chat", "mignon", "sur", "le"]

</details>

---

**Exercice 2 : CBOW vs Skip-gram**

Phrase : "le chien aboie"

**Pour CBOW :**
- Entrée ?
- Sortie attendue ?

**Pour Skip-gram :**
- Entrée ?
- Sortie attendue ?

<details>
<summary>Solution</summary>

**CBOW (fenêtre=1) :**
- Entrée : ["le", "aboie"]
- Sortie : "chien"

**Skip-gram (fenêtre=1) :**
- Entrée : "chien"
- Sortie : ["le", "aboie"]

</details>

---

**Exercice 3 : Analogies**

Si Word2Vec apprend bien, on devrait avoir :

```
roi - homme + femme ≈ reine
```

Pourquoi ? Quelle propriété mathématique cela implique-t-il ?

<details>
<summary>Réponse</summary>

**Explication :**

```
vec(roi) - vec(homme) ≈ vec(reine) - vec(femme)

→ Le vecteur "royauté" est capturé !

vec(roi) - vec(homme) ≈ vecteur de "royauté masculine"
vec(reine) - vec(femme) ≈ vecteur de "royauté féminine"
```

Propriété : Les relations sémantiques sont des translations dans l'espace vectoriel.

</details>

---

### JOUR 2 : Utiliser Word2Vec (gensim)

#### 💻 Premier contact avec gensim

```python
# word2vec_basics.py

from gensim.models import Word2Vec
import numpy as np

# Préparer les données
sentences = [
    ["le", "chat", "dort"],
    ["le", "chien", "aboie"],
    ["le", "chat", "mignon", "joue"],
    ["le", "chien", "court", "vite"],
    ["le", "chaton", "dort"],
]

# Entraîner Word2Vec
model = Word2Vec(
    sentences=sentences,
    vector_size=100,      # dimensionalité des vecteurs
    window=2,             # taille fenêtre de contexte
    min_count=1,          # ignorer mots avec freq < min_count
    workers=4,            # nombre de threads
    sg=1                  # 1=skip-gram, 0=CBOW
)

# Explorer le modèle
print("Vocabulaire:", len(model.wv))
print("Mots:", list(model.wv.index_to_key))

# Vecteur d'un mot
vec_chat = model.wv['chat']
print(f"\nVecteur 'chat': {vec_chat[:10]}...")  # premiers 10 dims

# Similarité entre mots
sim = model.wv.similarity('chat', 'chaton')
print(f"\nSimilarité chat-chaton: {sim:.3f}")

sim = model.wv.similarity('chat', 'dort')
print(f"Similarité chat-dort: {sim:.3f}")

# Mots les plus similaires
similars = model.wv.most_similar('chat', topn=3)
print(f"\nMots similaires à 'chat':")
for word, score in similars:
    print(f"  {word}: {score:.3f}")
```

---

#### 🎓 Entraîner sur un vrai corpus

```python
# word2vec_training.py

import re
from gensim.models import Word2Vec
from pathlib import Path

def load_and_preprocess_text(filepath):
    """
    Charge et prétraite un fichier texte
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Minuscules
    text = text.lower()
    
    # Remplacer ponctuation par espaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Séparer en phrases (simplifié)
    sentences = text.split('.')
    
    # Tokenizer chaque phrase
    tokenized = []
    for sent in sentences:
        words = sent.split()
        if len(words) > 3:  # ignorer phrases trop courtes
            tokenized.append(words)
    
    return tokenized


def train_word2vec(sentences, **params):
    """
    Entraîne Word2Vec avec paramètres personnalisés
    """
    default_params = {
        'vector_size': 100,
        'window': 5,
        'min_count': 5,
        'workers': 4,
        'sg': 1,        # skip-gram
        'epochs': 10
    }
    
    # Fusionner paramètres
    default_params.update(params)
    
    # Entraîner
    model = Word2Vec(sentences, **default_params)
    
    print(f"Vocabulaire: {len(model.wv)} mots")
    print(f"Corpus: {sum(len(s) for s in sentences)} mots total")
    
    return model


# Utilisation
if __name__ == "__main__":
    # Charger données
    # (Utilise n'importe quel fichier texte - roman, articles, etc.)
    sentences = load_and_preprocess_text("mon_corpus.txt")
    
    print(f"Chargé {len(sentences)} phrases")
    print(f"Exemple: {sentences[0]}")
    
    # Entraîner
    model = train_word2vec(
        sentences,
        vector_size=300,
        window=5,
        min_count=2
    )
    
    # Sauvegarder
    model.save("my_word2vec.model")
    
    # Tests
    test_words = ['chat', 'maison', 'rapide', 'beau']
    
    for word in test_words:
        if word in model.wv:
            similars = model.wv.most_similar(word, topn=5)
            print(f"\nMots similaires à '{word}':")
            for w, score in similars:
                print(f"  {w}: {score:.3f}")
```

---

#### 📚 Utiliser des modèles pré-entraînés

```python
# pretrained_word2vec.py

import gensim.downloader as api

# Télécharger un modèle pré-entraîné
# (première fois seulement, ~1.5 GB)
print("Téléchargement du modèle (patiente...)") 
model = api.load("word2vec-google-news-300")

print("Modèle chargé!")
print(f"Vocabulaire: {len(model)} mots")

# Tests
def explore_word(word):
    """
    Explore un mot
    """
    if word not in model:
        print(f"'{word}' pas dans le vocabulaire")
        return
    
    print(f"\n{'='*60}")
    print(f"Mot : {word}")
    print('='*60)
    
    # Similarités
    similars = model.most_similar(word, topn=10)
    print("\nMots similaires:")
    for w, score in similars:
        print(f"  {w:20s} {score:.3f}")
    
    # Analogies
    print("\nAnalogie: Paris est à France ce que Londres est à ___")
    try:
        result = model.most_similar(
            positive=['London', 'France'],
            negative=['Paris'],
            topn=1
        )
        print(f"  Réponse: {result[0][0]} (score: {result[0][1]:.3f})")
    except:
        print("  (pas trouvé)")


# Tests
explore_word('cat')
explore_word('king')
explore_word('python')

# Analogies célèbres
print("\n" + "="*60)
print("ANALOGIES CÉLÈBRES")
print("="*60)

analogies = [
    (['king', 'woman'], ['man'], "roi - homme + femme = ?"),
    (['Paris', 'Germany'], ['France'], "Paris - France + Germany = ?"),
    (['good', 'worst'], ['best'], "good - best + worst = ?"),
]

for positive, negative, description in analogies:
    try:
        result = model.most_similar(
            positive=positive,
            negative=negative,
            topn=1
        )
        print(f"\n{description}")
        print(f"  → {result[0][0]} (score: {result[0][1]:.3f})")
    except:
        print(f"\n{description}")
        print("  → Impossible de calculer")
```

---

#### ✅ Checkpoint Jour 2

Tu dois avoir :

- ✓ Installé gensim (`pip install gensim`)
- ✓ Entraîné un Word2Vec simple
- ✓ Testé un modèle pré-entraîné
- ✓ Exploré similarités et analogies
- ✓ Compris la magie des embeddings !

---

### JOUR 3-4 : Moteur de recherche v0.3

#### 💻 Intégrer Word2Vec dans le moteur

**Défi :** Comment représenter un document avec Word2Vec ?

**Problème :** Word2Vec donne des vecteurs pour des *mots*, pas des *documents* !

**Solutions :**

1. **Moyenne des vecteurs des mots**
2. **Moyenne pondérée par TF-IDF**
3. **Doc2Vec** (extension de Word2Vec)

```python
# search_engine_v03.py

import numpy as np
from gensim.models import Word2Vec

class Word2VecSearchEngine:
    def __init__(self, documents, w2v_model):
        """
        Moteur de recherche avec Word2Vec
        
        Args:
            documents: liste de strings
            w2v_model: modèle Word2Vec entraîné
        """
        self.documents = documents
        self.model = w2v_model
        
        # Pré-calculer les vecteurs des documents
        self.doc_vectors = [
            self.document_to_vector(doc) 
            for doc in documents
        ]
    
    def document_to_vector(self, document):
        """
        Transforme un document en vecteur (moyenne des mots)
        
        Args:
            document: string
        
        Returns:
            numpy array (vecteur du document)
        """
        words = document.lower().split()
        
        # Récupérer vecteurs des mots dans le vocabulaire
        word_vectors = []
        for word in words:
            if word in self.model.wv:
                word_vectors.append(self.model.wv[word])
        
        # Moyenne
        if len(word_vectors) == 0:
            # Document vide ou aucun mot connu
            return np.zeros(self.model.vector_size)
        
        return np.mean(word_vectors, axis=0)
    
    def document_to_vector_tfidf_weighted(self, document, tfidf_scores):
        """
        Transforme un document en vecteur (moyenne pondérée par TF-IDF)
        
        Args:
            document: string
            tfidf_scores: dict {word: tfidf_score}
        
        Returns:
            numpy array
        """
        words = document.lower().split()
        
        weighted_vectors = []
        total_weight = 0
        
        for word in words:
            if word in self.model.wv:
                weight = tfidf_scores.get(word, 1.0)
                weighted_vectors.append(self.model.wv[word] * weight)
                total_weight += weight
        
        if total_weight == 0:
            return np.zeros(self.model.vector_size)
        
        return np.sum(weighted_vectors, axis=0) / total_weight
    
    def search(self, query, top_k=5):
        """
        Recherche avec Word2Vec
        
        Args:
            query: string
            top_k: nombre de résultats
        
        Returns:
            liste de (doc_index, score)
        """
        # Vectoriser la requête
        query_vector = self.document_to_vector(query)
        
        # Calculer similarités
        similarities = []
        for i, doc_vec in enumerate(self.doc_vectors):
            # Similarité cosinus
            sim = np.dot(query_vector, doc_vec) / (
                np.linalg.norm(query_vector) * np.linalg.norm(doc_vec) + 1e-8
            )
            similarities.append((i, sim))
        
        # Trier
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def visualize_query(self, query):
        """
        Visualise comment la requête est comprise
        """
        print(f"\n{'='*60}")
        print(f"Analyse de : '{query}'")
        print('='*60)
        
        words = query.lower().split()
        
        print("\nMots de la requête:")
        for word in words:
            if word in self.model.wv:
                # Mots similaires
                similars = self.model.wv.most_similar(word, topn=3)
                print(f"  {word}:")
                for sim_word, score in similars:
                    print(f"    → {sim_word} ({score:.2f})")
            else:
                print(f"  {word}: [inconnu]")


# Test
if __name__ == "__main__":
    # Corpus
    documents = [
        "Le chat dort paisiblement sur le canapé",
        "Le chien aboie fort dans le jardin",
        "Le petit chaton joue avec une pelote",
        "Les chiens courent dans le parc",
        "Le chaton mange ses croquettes",
        "Python est un langage de programmation",
        "Java est utilisé pour développer",
        "Le machine learning utilise des algorithmes",
        "Les réseaux de neurones sont puissants",
        "Le deep learning révolutionne l'IA"
    ]
    
    # Préparer pour Word2Vec
    sentences = [doc.lower().split() for doc in documents]
    
    # Entraîner Word2Vec
    print("Entraînement de Word2Vec...")
    w2v_model = Word2Vec(
        sentences,
        vector_size=100,
        window=3,
        min_count=1,
        sg=1,
        epochs=100  # plus d'epochs pour petit corpus
    )
    
    # Créer moteur
    engine = Word2VecSearchEngine(documents, w2v_model)
    
    # Tests
    test_queries = [
        "kitten",      # synonyme de "chaton" en anglais !
        "feline",      # mot générique pour félins
        "coding",      # synonyme de "programmation"
        "neural"       # lié à "neurones"
    ]
    
    for query in test_queries:
        engine.visualize_query(query)
        
        results = engine.search(query, top_k=3)
        print("\nRésultats:")
        for rank, (idx, score) in enumerate(results, 1):
            print(f"{rank}. [{score:.3f}] {documents[idx]}")
        
        input("\nEntrée pour continuer...")
```

---

#### 🎯 Extension : Doc2Vec

**Idée :** Au lieu de faire la moyenne des mots, apprendre directement des vecteurs de documents !

```python
# doc2vec_engine.py

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class Doc2VecSearchEngine:
    def __init__(self, documents):
        """
        Moteur avec Doc2Vec
        """
        self.documents = documents
        
        # Préparer données (chaque doc a un tag unique)
        tagged_docs = [
            TaggedDocument(words=doc.lower().split(), tags=[str(i)])
            for i, doc in enumerate(documents)
        ]
        
        # Entraîner Doc2Vec
        print("Entraînement de Doc2Vec...")
        self.model = Doc2Vec(
            tagged_docs,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4,
            epochs=40
        )
        
        print("Entraîné!")
    
    def search(self, query, top_k=5):
        """
        Recherche avec Doc2Vec
        """
        # Inférer vecteur de la requête
        query_vector = self.model.infer_vector(query.lower().split())
        
        # Trouver documents similaires
        similar_docs = self.model.dv.most_similar([query_vector], topn=top_k)
        
        # Convertir tags en indices
        results = []
        for tag, score in similar_docs:
            idx = int(tag)
            results.append((idx, score))
        
        return results


# Comparaison Word2Vec vs Doc2Vec
if __name__ == "__main__":
    # (même corpus)
    
    print("\n=== Word2Vec ===")
    w2v_engine = Word2VecSearchEngine(documents, w2v_model)
    w2v_results = w2v_engine.search("kitten programming", top_k=3)
    
    for rank, (idx, score) in enumerate(w2v_results, 1):
        print(f"{rank}. [{score:.3f}] {documents[idx][:50]}...")
    
    print("\n=== Doc2Vec ===")
    d2v_engine = Doc2VecSearchEngine(documents)
    d2v_results = d2v_engine.search("kitten programming", top_k=3)
    
    for rank, (idx, score) in enumerate(d2v_results, 1):
        print(f"{rank}. [{score:.3f}] {documents[idx][:50]}...")
```

---

#### ✅ Checkpoint Jour 3-4

Tu dois avoir :

- ✓ Moteur de recherche Word2Vec
- ✓ Testé avec différentes requêtes
- ✓ Observé la magie des synonymes
- ✓ (Optionnel) Testé Doc2Vec
- ✓ Comparé avec TF-IDF

**Observations clés :**
- Word2Vec comprend les synonymes !
- Mais peut donner des résultats "trop généraux"
- TF-IDF est plus précis pour mots-clés exacts
- → Idée : combiner les deux ! (jour suivant)

---

### JOUR 5 : Visualisation et compréhension

#### 📊 Visualiser l'espace vectoriel

```python
# visualize_embeddings.py

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

def visualize_words_2d(model, words, method='pca'):
    """
    Visualise des mots en 2D
    
    Args:
        model: modèle Word2Vec
        words: liste de mots à visualiser
        method: 'pca' ou 'tsne'
    """
    # Récupérer vecteurs
    vectors = []
    valid_words = []
    
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
            valid_words.append(word)
    
    vectors = np.array(vectors)
    
    # Réduction de dimensionnalité
    if method == 'pca':
        reducer = PCA(n_components=2)
        coords_2d = reducer.fit_transform(vectors)
        title = "PCA"
    else:
        reducer = TSNE(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(vectors)
        title = "t-SNE"
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.5)
    
    # Annoter
    for i, word in enumerate(valid_words):
        plt.annotate(
            word,
            xy=(coords_2d[i, 0], coords_2d[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10
        )
    
    plt.title(f"Visualisation des mots ({title})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_similarity_matrix(model, words):
    """
    Heatmap des similarités entre mots
    """
    # Matrice de similarité
    n = len(words)
    sim_matrix = np.zeros((n, n))
    
    valid_words = []
    for i, word1 in enumerate(words):
        if word1 not in model.wv:
            continue
        valid_words.append(word1)
        
        for j, word2 in enumerate(words):
            if word2 not in model.wv:
                continue
            
            sim_matrix[i, j] = model.wv.similarity(word1, word2)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(label='Similarité')
    
    # Labels
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    plt.yticks(range(len(words)), words)
    
    # Annotations
    for i in range(n):
        for j in range(n):
            text = plt.text(
                j, i, f'{sim_matrix[i, j]:.2f}',
                ha="center", va="center", color="black", fontsize=8
            )
    
    plt.title("Matrice de similarité")
    plt.tight_layout()
    plt.show()


def explore_word_neighborhood(model, word, topn=20):
    """
    Explore le voisinage d'un mot
    """
    if word not in model.wv:
        print(f"'{word}' pas dans le vocabulaire")
        return
    
    # Mots similaires
    similars = model.wv.most_similar(word, topn=topn)
    similar_words = [word] + [w for w, _ in similars]
    
    # Visualiser
    visualize_words_2d(model, similar_words, method='tsne')


# Tests
if __name__ == "__main__":
    # Charger modèle
    model = Word2Vec.load("my_word2vec.model")
    
    # Test 1 : Comparer animaux, émotions, actions
    test_words = [
        'chat', 'chien', 'chaton',
        'heureux', 'triste', 'joyeux',
        'courir', 'marcher', 'sauter'
    ]
    
    visualize_words_2d(model, test_words, method='pca')
    visualize_similarity_matrix(model, test_words)
    
    # Test 2 : Explorer un mot
    explore_word_neighborhood(model, 'chat', topn=15)
```

---

#### 🔬 Analyser les analogies

```python
# analyze_analogies.py

def test_analogy(model, a, b, c, expected=None):
    """
    Teste une analogie : a - b + c ≈ ?
    
    Args:
        model: Word2Vec model
        a, b, c: mots de l'analogie
        expected: réponse attendue (optionnel)
    """
    try:
        result = model.wv.most_similar(
            positive=[a, c],
            negative=[b],
            topn=5
        )
        
        print(f"\n{a} - {b} + {c} = ?")
        print("-" * 40)
        
        for i, (word, score) in enumerate(result, 1):
            marker = "✓" if word == expected else " "
            print(f"{marker} {i}. {word:15s} (score: {score:.3f})")
        
        if expected:
            # Vérifier si présent dans top 5
            top_words = [w for w, _ in result]
            if expected in top_words:
                rank = top_words.index(expected) + 1
                print(f"\n✓ Réponse attendue trouvée (rang {rank})")
            else:
                print(f"\n✗ Réponse attendue pas dans le top 5")
    
    except Exception as e:
        print(f"\n{a} - {b} + {c} = ?")
        print(f"Erreur : {e}")


def comprehensive_analogy_test(model):
    """
    Batterie de tests d'analogies
    """
    analogies = [
        # Genre
        ('roi', 'homme', 'femme', 'reine'),
        ('acteur', 'homme', 'femme', 'actrice'),
        
        # Pays-Capitale
        ('Paris', 'France', 'Allemagne', 'Berlin'),
        ('Tokyo', 'Japon', 'Chine', 'Pékin'),
        
        # Verbes (temps)
        ('aller', 'va', 'faire', 'fait'),
        ('être', 'est', 'avoir', 'a'),
        
        # Comparatifs
        ('bon', 'meilleur', 'mauvais', 'pire'),
        ('grand', 'plus_grand', 'petit', 'plus_petit'),
    ]
    
    print("="*60)
    print("TESTS D'ANALOGIES")
    print("="*60)
    
    for a, b, c, expected in analogies:
        test_analogy(model, a, b, c, expected)
        input("\nAppuyez sur Entrée...")


# Test
if __name__ == "__main__":
    # Charger modèle pré-entraîné
    import gensim.downloader as api
    model = api.load("word2vec-google-news-300")
    
    comprehensive_analogy_test(model)
```

---

#### ✅ Checkpoint Jour 5

Tu dois avoir :

- ✓ Visualisé des embeddings en 2D
- ✓ Compris la structure de l'espace vectoriel
- ✓ Testé des analogies
- ✓ Vu les limites de Word2Vec sur petit corpus

---

### JOUR 6-7 : Récapitulatif et questions

#### 🤔 Questions de compréhension

**Question 1 : Pourquoi Word2Vec capture-t-il le sens ?**

<details>
<summary>Réponse</summary>

**Hypothèse distributionnelle :**
- Des mots dans des contextes similaires ont des sens similaires
- Word2Vec apprend à prédire les contextes
- Les vecteurs résultants encodent donc les similarités contextuelles
- Ces similarités contextuelles reflètent les similarités sémantiques

</details>

---

**Question 2 : Quand utiliser TF-IDF vs Word2Vec ?**

<details>
<summary>Réponse</summary>

**TF-IDF :**
- Recherche de mots-clés précis
- Documents techniques, légaux
- Besoin d'interprétabilité
- Contraintes de vitesse

**Word2Vec :**
- Recherche sémantique
- Comprendre synonymes
- Peu de mots-clés exacts
- Focus sur le sens

**Hybride (meilleur) :**
- Combine les deux forces
- Utilise TF-IDF pour filtrage initial
- Word2Vec pour raffinement sémantique

</details>

---

**Question 3 : Quelles limites restent même avec Word2Vec ?**

<details>
<summary>Réponse</summary>

**Limites :**
- Ordre des mots toujours perdu !
- Polysémie non gérée (un mot = un seul vecteur)
- Contexte du document ignoré
- Nécessite beaucoup de données d'entraînement

**Solutions (semaines suivantes) :**
- RNN/LSTM pour l'ordre
- ELMo/BERT pour contexte et polysémie
- Transformers pour tout ! 🚀

</details>

---

### JOUR 3-5 : Projet final intégrateur

#### 💻 Grand projet : Moteur de recherche hybride

**Objectif :** Combiner les forces de chaque méthode

```python
# hybrid_search_engine.py

class HybridSearchEngine:
    """
    Moteur qui combine TF-IDF (mots-clés) et Word2Vec (sens)
    """
    
    def __init__(self, documents, w2v_model):
        self.documents = documents
        
        # Créer les deux moteurs
        self.tfidf_engine = TfidfSearchEngine(documents)
        self.w2v_engine = Word2VecSearchEngine(documents, w2v_model)
    
    def search(self, query, top_k=5, alpha=0.5):
        """
        Recherche hybride
        
        Args:
            query: requête
            top_k: nombre de résultats
            alpha: poids TF-IDF (0=tout W2V, 1=tout TF-IDF)
        
        Returns:
            résultats fusionnés
        """
        # Recherche TF-IDF
        tfidf_results = self.tfidf_engine.search(query, top_k=len(self.documents))
        tfidf_scores = {idx: score for idx, score in tfidf_results}
        
        # Recherche Word2Vec
        w2v_results = self.w2v_engine.search(query, top_k=len(self.documents))
        w2v_scores = {idx: score for idx, score in w2v_results}
        
        # Fusionner les scores
        final_scores = {}
        for idx in range(len(self.documents)):
            tfidf_score = tfidf_scores.get(idx, 0)
            w2v_score = w2v_scores.get(idx, 0)
            
            # Combinaison pondérée
            final_score = alpha * tfidf_score + (1 - alpha) * w2v_score
            final_scores[idx] = final_score
        
        # Trier et retourner top_k
        sorted_results = sorted(
            final_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_results[:top_k]
    
    def search_with_explanation(self, query, top_k=3, alpha=0.5):
        """
        Recherche avec explication détaillée des scores
        """
        # Scores individuels
        tfidf_results = self.tfidf_engine.search(query, top_k=top_k)
        w2v_results = self.w2v_engine.search(query, top_k=top_k)
        hybrid_results = self.search(query, top_k=top_k, alpha=alpha)
        
        print("\n" + "="*70)
        print(f"REQUÊTE : '{query}' (alpha={alpha})")
        print("="*70)
        
        for rank, (doc_idx, final_score) in enumerate(hybrid_results, 1):
            # Trouver les scores individuels
            tfidf_score = next((s for i, s in tfidf_results if i == doc_idx), 0)
            w2v_score = next((s for i, s in w2v_results if i == doc_idx), 0)
            
            print(f"\n{rank}. Score final: {final_score:.4f}")
            print(f"   TF-IDF: {tfidf_score:.4f} | Word2Vec: {w2v_score:.4f}")
            print(f"   {self.documents[doc_idx][:80]}...")
        
        print("="*70)


# Test du moteur hybride
if __name__ == "__main__":
    # (même corpus que avant)
    
    # Créer moteur hybride
    hybrid_engine = HybridSearchEngine(documents, w2v_model)
    
    # Tester avec différents alpha
    query = "kitten"
    
    print("\nAlpha = 0 (100% Word2Vec)")
    hybrid_engine.search_with_explanation(query, alpha=0.0)
    
    print("\nAlpha = 0.5 (50/50)")
    hybrid_engine.search_with_explanation(query, alpha=0.5)
    
    print("\nAlpha = 1.0 (100% TF-IDF)")
    hybrid_engine.search_with_explanation(query, alpha=1.0)
```

---

#### 🎨 Extensions à implémenter (choisir 2-3)

**Extension 1 : Interface utilisateur**

```python
import streamlit as st

# Interface web pour le moteur de recherche
# Permet de tester facilement différentes requêtes
```

---

**Extension 2 : Analyse de pertinence**

```python
# Créer un jeu de test :
# - 10 requêtes
# - Pour chaque requête, les "bons" documents attendus
# - Mesurer la précision de chaque moteur
```

---

**Extension 3 : Support multilingue**

```python
# Utiliser des modèles Word2Vec pour plusieurs langues
# Tester requêtes en français, anglais, etc.
```

---

**Extension 4 : Visualisation**

```python
# Créer des graphiques comparant les 3 méthodes
# Scatter plot des documents dans l'espace 2D
# Heatmap des similarités
```

---

### JOUR 6-7 : Documentation et réflexion

#### 📝 Créer un document récapitulatif

**Structure suggérée :**

1. **Introduction**
   - Qu'est-ce que la représentation de texte ?
   - Pourquoi c'est important ?

2. **Bag of Words**
   - Principe
   - Implémentation
   - Avantages / Limites
   - Exemple de code

3. **TF-IDF**
   - Amélioration de BoW
   - Formules
   - Implémentation
   - Comparaison avec BoW

4. **Word2Vec**
   - Révolution des embeddings
   - Principe
   - Entraînement
   - Magie des analogies
   - Comparaison avec TF-IDF

5. **Synthèse**
   - Quand utiliser quoi ?
   - Limites restantes
   - Ouverture vers BERT (semaine suivante)

---

#### 🤔 Questions de réflexion finale

**Question 1 :** Si tu devais expliquer Word2Vec à quelqu'un sans background technique, comment ferais-tu ?

**Question 2 :** Quelles sont les 3 choses les plus importantes que tu as apprises ?

**Question 3 :** Quelles questions te restes-tu ?

**Question 4 :** Es-tu prêt pour la suite (Transformers, LLM) ?

---

## 🎓 BILAN FINAL DE LA PHASE 1

### ✅ Ce que tu dois maîtriser maintenant

**Concepts théoriques**

- ✓ Bag of Words (comptage de mots)
- ✓ TF-IDF (valorisation des mots rares)
- ✓ Word embeddings (vecteurs denses sémantiques)
- ✓ Similarité cosinus (mesure de proximité)
- ✓ Hypothèse distributionnelle

**Compétences pratiques**

- ✓ Coder BoW de A à Z
- ✓ Coder TF-IDF de A à Z
- ✓ Utiliser sklearn pour le NLP
- ✓ Utiliser Word2Vec (gensim)
- ✓ Entraîner un modèle Word2Vec
- ✓ Construire un moteur de recherche

**Projets réalisés**

- ✓ Moteur v0.1 (BoW)
- ✓ Moteur v0.2 (TF-IDF)
- ✓ Moteur v0.3 (Word2Vec)
- ✓ Moteur hybride (combinaison)

---

### 📊 Auto-évaluation

Note-toi de 1 à 5 sur chaque point :

| Compétence | Note /5 | Commentaire |
|------------|---------|-------------|
| Je comprends BoW | | |
| Je comprends TF-IDF | | |
| Je comprends Word2Vec | | |
| Je peux coder un moteur from scratch | | |
| Je peux utiliser sklearn/gensim | | |
| Je peux expliquer à quelqu'un d'autre | | |

Si tu as < 4 sur un point, revois cette partie !

---

## 🚀 PROCHAINES ÉTAPES

### Phase 2 : Réseaux de neurones

- Comprendre comment un neurone fonctionne
- Construire un réseau de neurones
- Comprendre la backpropagation
- Entraîner sur du texte

### Phase 3 : Réseaux récurrents et attention

- RNN / LSTM pour les séquences
- Mécanisme d'attention
- Comprendre pourquoi c'est plus puissant que Word2Vec

### Phase 4 : Transformers et LLM

- Architecture Transformer
- BERT, GPT
- Comment ChatGPT fonctionne

---

## 📚 RESSOURCES COMPLÉMENTAIRES

### Vidéos recommandées

- StatQuest : "Word2Vec" (très visuel)
- 3Blue1Brown : "Neural Networks" (pour la suite)

### Lectures

- Article original Word2Vec : "Efficient Estimation of Word Representations in Vector Space" (Mikolov et al., 2013)

### Datasets pour pratiquer

- Wikipedia dumps : Textes en français/anglais
- Common Crawl : Web crawl géant
- Kaggle datasets : Nombreux corpus annotés

### Outils

- **gensim** : Word2Vec, Doc2Vec
- **sklearn** : TF-IDF, préprocessing
- **NLTK** : Traitement du langage
- **spaCy** : NLP production-ready

---

## 💡 CONSEILS FINAUX

**1. Pratique > Théorie**

Tu dois avoir codé chaque concept, pas juste lu dessus.

**2. Compare toujours**

À chaque nouvelle méthode, compare avec la précédente.

**3. Utilise tes propres données**

C'est plus motivant et tu verras les vrais problèmes !

**4. Ne reste pas bloqué**

Si tu ne comprends pas quelque chose après 2h, passe à la suite et reviens plus tard avec un esprit frais.

**5. Enseigne ce que tu apprends**

Explique à quelqu'un (ou écris un article). C'est le meilleur test de compréhension.

**6. Sois patient**

Word2Vec peut sembler magique, mais c'est normal de ne pas tout comprendre du premier coup.

**7. Amusez-toi !**

Teste des trucs bizarres, fais des erreurs, casse des trucs. C'est comme ça qu'on apprend le mieux !

---

## 🎉 FÉLICITATIONS !

Si tu as suivi cette roadmap, tu as des bases **SOLIDES** !

Tu es prêt pour :

- ✅ Comprendre les papiers de recherche en NLP
- ✅ Attaquer les réseaux de neurones
- ✅ Comprendre comment fonctionnent les LLM
- ✅ Construire tes propres projets NLP

**Bon courage pour la suite !** 🚀

---

Questions ? Besoin d'aide sur une partie ? N'hésite pas à revenir vers moi ! 😊
