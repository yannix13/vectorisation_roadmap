# 🗺️ ROADMAP DÉTAILLÉE : PHASE 1 - Représenter du texte

**Objectif final :** Comprendre comment transformer du texte en vecteurs et construire un moteur de recherche qui s'améliore progressivement.

**Durée totale :** 3-4 semaines (flexible selon ton rythme)

---

<details>
<summary><strong>📅 SEMAINE 1 : Bag of Words (BoW)</strong></summary>

### 🎯 Objectif de la semaine
Comprendre la méthode la plus simple pour transformer texte → vecteur

---

<details>
<summary>JOUR 1 : Théorie + Compréhension conceptuelle</summary>

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

</details>

---

<details>
<summary>JOUR 2-3 : Implémentation de base (à la main)</summary>

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

</details>

---

<details>
<summary>JOUR 4-5 : Mini moteur de recherche v0.1</summary>

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

</details>

---

<details>
<summary>JOUR 6-7 : Optimisation et expérimentation</summary>

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

</details>

---

### ✅ Checkpoint Semaine 1

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

</details>

---

<details>
<summary><strong>📅 SEMAINE 2 : TF-IDF</strong></summary>

### 🎯 Objectif de la semaine

Comprendre comment valoriser les mots importants et réduire l'impact des mots courants

---

<details>
<summary>JOUR 1 : Comprendre TF-IDF</summary>

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

</details>

---

<details>
<summary>JOUR 2-3 : Implémentation de TF-IDF</summary>

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

</details>

---

<details>
<summary>JOUR 4-5 : Moteur de recherche v0.2</summary>

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

</details>

---

<details>
<summary>JOUR 6-7 : sklearn et optimisations</summary>

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

</details>

---

### ✅ Checkpoint Semaine 2

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

</details>

---

<details>
<summary><strong>🆕 SEMAINE 2.5 (AJOUT) : Du one-hot à la couche d'embedding</strong></summary>

> ### 🧩 Note : pourquoi ce module a été ajouté
>
> **Ce module n'était pas dans la roadmap d'origine — on l'a ajouté ensemble.**
>
> En relisant la roadmap, on a constaté trois chaînons manquants entre TF-IDF (Semaine 2) et Word2Vec (Semaine 3) :
> 1. Le **one-hot encoding** n'était jamais nommé, alors que c'est la brique atomique sur laquelle reposent BoW *et* les embeddings.
> 2. La **couche d'embedding** (la matrice de poids *entraînable*) n'apparaissait nulle part : Word2Vec était utilisé comme une boîte noire `gensim` sans jamais ouvrir le capot.
> 3. Du coup, le moment **« j'entraîne ma première couche à la main »** (gradient descent, une seule couche) n'existait pas.
>
> Ce module comble exactement ce trou. Il transforme le passage « TF-IDF → Word2Vec magique » en une progression continue : **one-hot → couche d'embedding → entraînement à la main → *c'est ça, Word2Vec*.**

---

### 🎯 Objectif de la semaine

Comprendre **comment** on passe d'un vecteur creux (one-hot/TF-IDF) à un vecteur dense appris — en codant soi-même une couche d'embedding et en l'entraînant à la main. À la fin, `gensim.Word2Vec(...)` ne sera plus une boîte noire.

---

<details>
<summary>JOUR 1 : One-hot encoding — la brique atomique</summary>

#### 📖 Concepts à maîtriser

**1. Qu'est-ce que le one-hot encoding ?**

C'est la représentation la plus élémentaire d'**un seul mot** : un vecteur de la taille du vocabulaire, rempli de zéros, avec **un seul 1** à la position du mot.

```
Vocabulaire : ["chat", "chaton", "chien", "dort", "aboie"]
               0       1         2        3       4

"chat"   → [1, 0, 0, 0, 0]
"chaton" → [0, 1, 0, 0, 0]
"chien"  → [0, 0, 1, 0, 0]
```

**2. Le lien que personne ne t'a montré : BoW = somme de one-hots**

```
"le chat dort"  (on ignore "le")
  chat → [1, 0, 0, 0, 0]
  dort → [0, 0, 0, 1, 0]
  ─────────────────────── somme
  BoW  → [1, 0, 0, 1, 0]
```

> 💡 **Le one-hot est donc l'atome.** BoW n'est qu'une addition de one-hots, et TF-IDF une version pondérée. C'est pour ça qu'il vient *avant* BoW dans l'ordre logique, même si on l'apprend après.

**3. Le problème fondamental : tous les one-hots sont orthogonaux**

```python
cosine("chat", "chaton")  # [1,0,0,0,0] · [0,1,0,0,0] = 0
cosine("chat", "chien")   # = 0
cosine("chat", "avion")   # = 0
```

**Tous les mots sont à égale distance les uns des autres.** « chat » n'est pas plus proche de « chaton » que de « avion ». Le one-hot ne contient **aucune information de sens** — juste une identité.

> 🔑 **C'est LA motivation de tout ce qui suit.** On veut remplacer ces vecteurs creux et orthogonaux par des vecteurs **denses** où « chat » et « chaton » seraient *proches*. Ce passage, c'est la couche d'embedding (Jour 2).

---

#### 🔤 Au passage : tokeniser ≠ vectoriser

Profite de ce module pour bien ancrer la distinction :

```
"le chat dort"
      │
      ▼  TOKENISER  →  découper en unités + attribuer un ID
  ["le"→0, "chat"→1, "dort"→2]
      │
      ▼  VECTORISER  →  transformer chaque ID en vecteur
  one-hot, BoW, TF-IDF, embedding...
```

Le **tokeniser** produit des *indices* (des entiers). Le **vectoriser** transforme ces indices en *vecteurs*. Le one-hot est le premier pont entre les deux : `indice 1 → [0,1,0,...]`.

---

#### ✏️ Exercices papier (20-30 min)

**Exercice 1 :** Vocabulaire `["roi", "reine", "homme", "femme"]`. Écris le one-hot de chaque mot.

<details>
<summary>Solution</summary>

```
roi    → [1, 0, 0, 0]
reine  → [0, 1, 0, 0]
homme  → [0, 0, 1, 0]
femme  → [0, 0, 0, 1]
```
</details>

**Exercice 2 :** Calcule `cosine(roi, reine)` avec ces one-hots. Que vaut-il ? Est-ce satisfaisant ?

<details>
<summary>Solution</summary>

Produit scalaire = 0 → **cosine = 0**. Insatisfaisant : « roi » et « reine » sont sémantiquement très proches, mais le one-hot les voit aussi éloignés que deux mots au hasard.
</details>

---

#### 💻 Mini-implémentation

```python
# one_hot.py

def build_vocab(documents):
    words = sorted(set(w for doc in documents for w in doc.lower().split()))
    word2idx = {w: i for i, w in enumerate(words)}
    return word2idx

def one_hot(word, word2idx):
    vec = [0] * len(word2idx)
    if word in word2idx:
        vec[word2idx[word]] = 1
    return vec

if __name__ == "__main__":
    docs = ["chat chaton chien", "dort aboie"]
    vocab = build_vocab(docs)
    print(vocab)
    print("chat   →", one_hot("chat", vocab))
    print("chaton →", one_hot("chaton", vocab))
```

#### ✅ Checkpoint Jour 1

- ✓ Je sais écrire le one-hot d'un mot
- ✓ Je comprends que BoW = somme de one-hots
- ✓ Je peux expliquer pourquoi tous les one-hots sont orthogonaux
- ✓ Je distingue tokeniser (→ indices) et vectoriser (→ vecteurs)

</details>

---

<details>
<summary>JOUR 2 : La couche d'embedding = une matrice de poids</summary>

#### 📖 Concepts à maîtriser

**1. L'idée clé : remplacer le one-hot par une ligne de matrice**

Une **couche d'embedding** est simplement une matrice `W` de taille `(taille_vocabulaire × dimension_embedding)`. Chaque **ligne** est le vecteur dense d'un mot.

```
W  (5 mots × 3 dimensions) :

         dim0   dim1   dim2
chat   [ 0.21  -0.43   0.88 ]   ← ligne 0
chaton [ 0.19  -0.40   0.82 ]   ← ligne 1
chien  [-0.55   0.12   0.30 ]   ← ligne 2
dort   [ 0.71   0.05  -0.22 ]   ← ligne 3
aboie  [-0.33   0.60  -0.10 ]   ← ligne 4
```

**2. Pourquoi one-hot × W = la bonne ligne (le « lookup »)**

```
one-hot("chaton") · W  =  [0,1,0,0,0] · W  =  ligne 1 de W  =  [0.19, -0.40, 0.82]
```

> 🔑 **Multiplier un one-hot par la matrice, c'est juste *sélectionner une ligne*.** C'est pour ça qu'en pratique on ne fait jamais la multiplication : on fait `W[indice]` directement. Une couche d'embedding **EST** une table de correspondance `indice → vecteur` — mais une table dont les valeurs sont **apprises**.

**3. La grande bascule conceptuelle**

| | One-hot / TF-IDF | Couche d'embedding |
|---|---|---|
| Dimension | = taille vocab (creux) | petite, ex. 3-300 (dense) |
| Valeurs | fixes (0/1 ou comptes) | **apprises** par entraînement |
| Sémantique | aucune (orthogonaux) | proximité = sens |
| D'où viennent les nombres ? | règles | **gradient descent** (Jour 3-4) |

---

#### ✏️ Exercice

Avec la matrice `W` ci-dessus, donne sans calcul le vecteur de « dort ». Puis calcule `cosine(chat, chaton)` vs `cosine(chat, chien)`.

<details>
<summary>Solution</summary>

`dort` = ligne 3 = `[0.71, 0.05, -0.22]` (simple lookup).

`chat`=[0.21,-0.43,0.88], `chaton`=[0.19,-0.40,0.82] → très alignés → cosine ≈ **0.99**.
`chien`=[-0.55,0.12,0.30] → cosine(chat,chien) ≈ **-0.2**.

→ Cette fois, « chat » est bien plus proche de « chaton » que de « chien ». **C'est exactement ce que le one-hot ne pouvait pas faire.** Reste à savoir *comment* obtenir ces nombres → Jour 3-4.
</details>

---

#### 💻 La couche d'embedding en numpy

```python
# embedding_layer.py
import numpy as np

class EmbeddingLayer:
    def __init__(self, vocab_size, embedding_dim, seed=0):
        rng = np.random.default_rng(seed)
        # W : une ligne par mot, initialisée au hasard (sera apprise)
        self.W = rng.normal(0, 0.1, size=(vocab_size, embedding_dim))

    def lookup(self, idx):
        # one-hot × W  ==  W[idx]  (on prend juste la ligne)
        return self.W[idx]

if __name__ == "__main__":
    emb = EmbeddingLayer(vocab_size=5, embedding_dim=3)
    print("Vecteur du mot d'indice 1 :", emb.lookup(1))
    print("Matrice complète :\n", emb.W)
```

#### ✅ Checkpoint Jour 2

- ✓ Une couche d'embedding est une matrice `(vocab × dim)`
- ✓ `one-hot × W` revient à sélectionner une ligne → le « lookup »
- ✓ Je comprends que ces nombres sont **appris**, pas fixés
- ✓ Je vois ce qui sépare un vecteur creux d'un vecteur dense

</details>

---

<details>
<summary>JOUR 3-4 : Entraîner la couche à la main (une seule couche)</summary>

#### 📖 L'idée : apprendre W avec l'hypothèse distributionnelle

> « Un mot est caractérisé par la compagnie qu'il tient. » — Firth

On va **entraîner** la matrice `W` pour qu'un mot serve à **prédire ses voisins**. C'est *exactement* le principe de Word2Vec (skip-gram), mais codé à la main, avec **une seule couche cachée**.

**Le modèle (le plus simple possible) :**

```
mot central (indice)
      │  lookup dans W_emb   ← LA couche d'embedding (ce qu'on veut apprendre)
      ▼
  vecteur dense  h  (dim = embedding_dim)
      │  × W_out  + softmax
      ▼
  probabilités sur tout le vocabulaire  →  « quels mots sont autour ? »
```

On compare la prédiction au **vrai voisin** (one-hot), on calcule l'erreur, et on corrige `W_emb` et `W_out` par **gradient descent**. Au fil des itérations, les mots qui partagent des contextes voient leurs lignes de `W_emb` se rapprocher.

> 🔑 `W_emb`, c'est ta couche d'embedding. Une fois l'entraînement fini, **on jette `W_out` et on garde `W_emb`** : ses lignes sont tes word embeddings.

---

#### 💻 Skip-gram from scratch (numpy, une couche)

```python
# train_embeddings.py
import numpy as np

# 1) Corpus jouet : "chat"/"chaton" partagent le contexte "dort",
#    "chien" partage "aboie" → on veut que chat ≈ chaton
corpus = [
    ["chat", "dort"], ["chaton", "dort"], ["chat", "dort"],
    ["chien", "aboie"], ["chien", "aboie"], ["chaton", "dort"],
]

# 2) Vocabulaire
words = sorted(set(w for s in corpus for w in s))
word2idx = {w: i for i, w in enumerate(words)}
V = len(words)

# 3) Paires (centre, contexte) dans les deux sens
pairs = []
for sent in corpus:
    for i, center in enumerate(sent):
        for j, context in enumerate(sent):
            if i != j:
                pairs.append((word2idx[center], word2idx[context]))

# 4) Paramètres : DEUX matrices
D = 2                     # dimension d'embedding (2 pour visualiser)
rng = np.random.default_rng(0)
W_emb = rng.normal(0, 0.1, (V, D))   # ← LA couche d'embedding (à garder)
W_out = rng.normal(0, 0.1, (D, V))   # ← couche de sortie (à jeter)
lr = 0.1

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

# 5) Boucle d'entraînement (gradient descent à la main)
for epoch in range(300):
    loss = 0.0
    for center, context in pairs:
        h = W_emb[center]              # lookup : one-hot × W_emb
        scores = h @ W_out             # une seule couche linéaire
        y = softmax(scores)            # probabilités sur le vocab

        loss -= np.log(y[context] + 1e-9)

        # cible one-hot du vrai voisin
        target = np.zeros(V); target[context] = 1
        dscores = y - target           # gradient de la cross-entropy + softmax

        # rétropropagation (chain rule)
        grad_Wout = np.outer(h, dscores)
        grad_h    = W_out @ dscores

        # mise à jour
        W_out          -= lr * grad_Wout
        W_emb[center]  -= lr * grad_h

    if epoch % 50 == 0:
        print(f"epoch {epoch:3d}  loss={loss/len(pairs):.3f}")

# 6) Résultat : les embeddings appris
def cos(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

print("\nEmbeddings appris :")
for w in words:
    print(f"  {w:7s} {W_emb[word2idx[w]].round(2)}")

print("\nchat ~ chaton :", round(cos(W_emb[word2idx['chat']], W_emb[word2idx['chaton']]), 3))
print("chat ~ chien  :", round(cos(W_emb[word2idx['chat']], W_emb[word2idx['chien']]), 3))
```

**Ce que tu dois observer :** après entraînement, `cos(chat, chaton)` est nettement plus élevé que `cos(chat, chien)` — alors qu'au départ (one-hot) ils valaient tous **0**. Tu viens d'apprendre du **sens** à partir de rien d'autre que des co-occurrences.

---

#### 🔬 Expériences à tenter

1. Mets `D = 2` et **trace** les 5 mots sur un plan (`matplotlib.scatter`) : tu verras « chat »/« chaton » se regrouper.
2. Change le corpus pour que « chien » partage aussi « dort » : observe son vecteur se rapprocher des chats.
3. Augmente/diminue `lr` et le nombre d'epochs : vois l'effet sur la loss.

#### ✅ Checkpoint Jour 3-4

- ✓ J'ai entraîné une matrice d'embedding à la main
- ✓ Je comprends le rôle de `W_emb` (à garder) vs `W_out` (à jeter)
- ✓ Je relie softmax + cross-entropy + gradient descent au résultat
- ✓ Je constate que des mots à contextes proches finissent proches

</details>

---

<details>
<summary>JOUR 5 : Le pont — « ce que tu viens de coder, c'est Word2Vec »</summary>

#### 🌉 Tout se recolle

Ce que tu as codé au Jour 3-4 **est** la version minimale de Word2Vec (skip-gram) :

| Ton code à la main | gensim `Word2Vec(...)` |
|---|---|
| `W_emb` (matrice apprise) | `model.wv.vectors` |
| lookup `W_emb[center]` | `model.wv["chat"]` |
| paires (centre, contexte) | fenêtre `window=...` |
| `D = 2` | `vector_size=100` |
| skip-gram à la main | `sg=1` |
| softmax sur tout le vocab | (optimisé : negative sampling) |

> 🔑 Quand tu écriras `Word2Vec(sentences, sg=1)` en Semaine 3, tu sauras **exactement** ce qu'il y a sous le capot : la même couche d'embedding, le même entraînement, juste en (beaucoup) plus optimisé et sur un (beaucoup) plus gros corpus.

#### 🔭 Et la suite ?

- **Polysémie / ordre des mots** : une couche d'embedding donne *un seul* vecteur par mot, fixe. Pour gérer l'ordre et le contexte, il faudra des modèles de séquence → **RNN/LSTM**, puis l'**attention** et les **Transformers** (phases suivantes).
- La couche d'embedding que tu viens de coder **reste la première couche** de tous ces modèles, y compris GPT. Tu ne la jettes jamais — tu construis dessus.

#### ✅ Checkpoint Jour 5

- ✓ Je sais relier mon code à la main à `gensim.Word2Vec`
- ✓ Je comprends que l'embedding layer est la fondation de tout le NLP moderne
- ✓ Je suis prêt à aborder Word2Vec (Semaine 3) sans boîte noire

</details>

---

### ✅ Checkpoint Semaine 2.5

**Compétences acquises :**

- ✓ One-hot encoding et son orthogonalité
- ✓ Couche d'embedding = matrice de poids apprise (lookup)
- ✓ Entraînement d'une couche à la main (gradient descent)
- ✓ Compréhension profonde de ce qu'est Word2Vec *avant* de l'utiliser

**Questions de réflexion :**

1. Pourquoi un one-hot ne peut-il jamais capturer la similarité entre deux mots ?
2. Que représente concrètement une *ligne* de la matrice d'embedding ?
3. Pourquoi jette-t-on `W_out` mais garde-t-on `W_emb` ?

</details>

---

<details>
<summary><strong>📅 SEMAINE 3 : Word2Vec</strong></summary>

### 🎯 Objectif de la semaine

Comprendre les word embeddings : représentations vectorielles denses qui capturent le sens des mots

---

<details>
<summary>JOUR 1 : Théorie des word embeddings</summary>

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

</details>

---

<details>
<summary>JOUR 2 : Utiliser Word2Vec (gensim)</summary>

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

</details>

---

<details>
<summary>JOUR 3-4 : Moteur de recherche v0.3</summary>

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
        """
        words = document.lower().split()
        
        # Récupérer vecteurs des mots dans le vocabulaire
        word_vectors = []
        for word in words:
            if word in self.model.wv:
                word_vectors.append(self.model.wv[word])
        
        # Moyenne
        if len(word_vectors) == 0:
            return np.zeros(self.model.vector_size)
        
        return np.mean(word_vectors, axis=0)
    
    def document_to_vector_tfidf_weighted(self, document, tfidf_scores):
        """
        Transforme un document en vecteur (moyenne pondérée par TF-IDF)
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
        """
        # Vectoriser la requête
        query_vector = self.document_to_vector(query)
        
        # Calculer similarités
        similarities = []
        for i, doc_vec in enumerate(self.doc_vectors):
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
                similars = self.model.wv.most_similar(word, topn=3)
                print(f"  {word}:")
                for sim_word, score in similars:
                    print(f"    → {sim_word} ({score:.2f})")
            else:
                print(f"  {word}: [inconnu]")


# Test
if __name__ == "__main__":
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

</details>

---

<details>
<summary>JOUR 5 : Visualisation et compréhension</summary>

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

</details>

---

<details>
<summary>JOUR 6-7 : Récapitulatif et questions</summary>

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

</details>

---

<details>
<summary>JOUR 3-5 : Projet final intégrateur</summary>

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
        tfidf_results = self.tfidf_engine.search(query, top_k=top_k)
        w2v_results = self.w2v_engine.search(query, top_k=top_k)
        hybrid_results = self.search(query, top_k=top_k, alpha=alpha)
        
        print("\n" + "="*70)
        print(f"REQUÊTE : '{query}' (alpha={alpha})")
        print("="*70)
        
        for rank, (doc_idx, final_score) in enumerate(hybrid_results, 1):
            tfidf_score = next((s for i, s in tfidf_results if i == doc_idx), 0)
            w2v_score = next((s for i, s in w2v_results if i == doc_idx), 0)
            
            print(f"\n{rank}. Score final: {final_score:.4f}")
            print(f"   TF-IDF: {tfidf_score:.4f} | Word2Vec: {w2v_score:.4f}")
            print(f"   {self.documents[doc_idx][:80]}...")
        
        print("="*70)


# Test du moteur hybride
if __name__ == "__main__":
    hybrid_engine = HybridSearchEngine(documents, w2v_model)
    
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

</details>

---

<details>
<summary>JOUR 6-7 : Documentation et réflexion</summary>

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

</details>

</details>

---

<details open>
<summary><strong>🆕 SEMAINE 5 (AJOUT) : Du word embedding au document embedding (et au RAG)</strong></summary>

> ### 🧩 Note : pourquoi ce module a été ajouté
>
> **Ce module n'était PAS dans la roadmap d'origine — on l'a ajouté ensemble, après coup.**
>
> La roadmap d'origine s'arrêtait à Word2Vec (Semaine 3-4), c'est-à-dire à des embeddings de **mots**. Or, pour aller vers ton objectif — **les apps agentiques et comprendre les LLM** — il manquait deux chaînons :
> 1. On sait vectoriser un **mot** (Word2Vec), mais pas encore un **document entier** proprement. Comment passe-t-on de « un vecteur par mot » à « un vecteur par phrase/document » ?
> 2. La brique centrale de toute app LLM moderne — le **RAG** (Retrieval-Augmented Generation) — n'apparaissait nulle part, alors qu'elle repose **exactement** sur ce que tu maîtrises déjà : transformer du texte en vecteur dense et chercher par **similarité cosinus**.
>
> Ce module comble ce trou. Il prolonge la progression naturelle : **word embedding → document embedding → recherche sémantique → RAG → c'est ça, la base d'une app agentique.**
>
> ⚠️ **Tout ici réutilise des notions que tu connais déjà** (vecteurs denses, cosinus, moteur de recherche v0.3). On ne fait qu'ajouter le dernier étage de la fusée.

---

### 🎯 Objectif de la semaine

Comprendre **comment** on obtient un seul vecteur dense pour un **document entier** (pas juste un mot), pourquoi la simple moyenne ne suffit pas, ce qu'est un **sentence embedding** (`sentence-transformers`), et comment ces vecteurs + la similarité cosinus que tu connais déjà forment le squelette d'un système **RAG**. À la fin, « envoyer mes documents à un LLM pour qu'il réponde dessus » ne sera plus une boîte noire.

---

<details>
<summary>JOUR 1 : Du mot au document — le problème du pooling</summary>

#### 📖 Concepts à maîtriser

**1. Le décalage : Word2Vec vectorise des MOTS, pas des documents**

À la fin de la Semaine 3, tu as un vecteur dense par **mot** :

```
"chat"  → [0.21, -0.43, 0.88, ...]   (dim 100)
"chien" → [0.19, -0.39, 0.81, ...]
"avion" → [-0.66, 0.12, -0.05, ...]
```

Mais un moteur de recherche compare des **documents** (des phrases, des paragraphes). Il te faut **un seul vecteur par document**. Comment fabriquer le vecteur d'`"un chat dort sur le canapé"` à partir des vecteurs de ses mots ?

**2. La solution naïve : la moyenne (mean pooling)**

On additionne les vecteurs de tous les mots du document et on divise par leur nombre :

```
vec(document) = moyenne( vec("chat"), vec("dort"), vec("canapé"), ... )
```

C'est **exactement** ce que faisait déjà ton moteur v0.3 en Semaine 3 (Jour 3-4) sans le nommer. Ce procédé s'appelle le **mean pooling** : « pooling » = agréger plusieurs vecteurs en un seul.

> 💡 Le mean pooling marche étonnamment bien comme **point de départ**. Un document parlant de cuisine aura un vecteur moyen proche d'autres documents de cuisine, car ils partagent beaucoup de mots au sens proche.

**3. Pourquoi la moyenne ne suffit pas : elle perd l'ORDRE et le contexte**

La moyenne est **commutative** : changer l'ordre des mots ne change pas le résultat.

```
"le chat mange la souris"   → même vecteur moyen que ↓
"la souris mange le chat"
```

Pourtant ces deux phrases veulent dire le **contraire**. La moyenne écrase aussi le **contexte** : le mot « avocat » (fruit vs métier) a un seul vecteur Word2Vec, donc la moyenne ne peut pas le désambiguïser selon la phrase.

> 🔑 **C'est LA motivation du Jour 2.** On veut un encodeur qui produit le vecteur du document en **tenant compte de l'ordre et du contexte** des mots — pas une simple moyenne. C'est ce que font les **sentence embeddings**.

#### 🎯 Exercice du jour

**Exercice :** Avec un modèle Word2Vec (gensim) déjà chargé, écris une fonction `document_vector(texte, model)` qui fait le mean pooling des mots présents dans le vocabulaire. Teste-la sur deux phrases proches et deux phrases éloignées, et calcule le cosinus (ta fonction de Semaine 1 !) entre elles.

<details>
<summary>Solution</summary>

```python
import numpy as np

def document_vector(texte, model):
    mots = [m for m in texte.lower().split() if m in model.wv]
    if not mots:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[m] for m in mots], axis=0)  # mean pooling

# cosine() = ta fonction de la Semaine 1
v1 = document_vector("le chat dort sur le canapé", model)
v2 = document_vector("un félin se repose sur le sofa", model)
v3 = document_vector("la bourse a chuté ce matin", model)
print(cosine(v1, v2))  # élevé : même sujet
print(cosine(v1, v3))  # faible : sujets différents
```

Vérifie ensuite que `document_vector("le chat mange la souris")` et `document_vector("la souris mange le chat")` donnent **le même vecteur** → c'est la limite à dépasser au Jour 2.

</details>

#### ✅ Mini-checkpoint Jour 1

- ✓ Je sais qu'un document a besoin d'**un seul** vecteur
- ✓ Je sais ce qu'est le **mean pooling** (et que mon moteur v0.3 le faisait déjà)
- ✓ Je peux expliquer pourquoi la moyenne perd l'**ordre** et le **contexte**

</details>

---

<details>
<summary>JOUR 2 : Les sentence embeddings (sentence-transformers / SBERT)</summary>

#### 📖 Concepts à maîtriser

**1. L'idée : un modèle entraîné à sortir directement le vecteur d'une phrase**

Plutôt que de moyenner des vecteurs de mots, on utilise un **modèle entraîné spécifiquement** pour lire une phrase entière et produire **un seul vecteur dense** qui capture son sens — en tenant compte de l'ordre et du contexte. C'est un **sentence embedding**.

Le plus connu : **SBERT** (Sentence-BERT), accessible via la librairie `sentence-transformers`.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")   # petit, rapide, gratuit
vec = model.encode("le chat mange la souris")      # → vecteur dense (dim 384)
```

> 🔑 **Le geste mental est identique à tout ce que tu as fait** : `texte → vecteur dense`. Word2Vec encodait un **mot** ; SBERT encode une **phrase entière**. Même finalité, encodeur plus puissant.

**2. Ce que ça change concrètement par rapport à la moyenne**

| | Word2Vec + moyenne | Sentence embedding (SBERT) |
|---|---|---|
| Unité encodée | mot (puis moyenné) | phrase entière, d'un coup |
| Ordre des mots | **perdu** | **pris en compte** |
| Mot ambigu (« avocat ») | 1 seul vecteur | désambiguïsé par le contexte |
| « chat mange souris » vs « souris mange chat » | même vecteur | **vecteurs différents** |

Cette fois, les deux phrases « le chat mange la souris » / « la souris mange le chat » ont bien des vecteurs **différents**. C'est exactement ce que le mean pooling ne pouvait pas faire.

**3. Le lien avec les LLM (ouverture)**

SBERT est construit sur un **Transformer** (BERT), la même famille d'architecture que les LLM. Tu n'as pas besoin d'en comprendre les détails maintenant — retiens juste : **un sentence embedding, c'est un encodeur de type LLM qui transforme un texte en un point dans un espace vectoriel sémantique.** Et dans cet espace, la proximité se mesure encore et toujours avec... le **cosinus**.

#### 🎯 Exercice du jour

**Exercice :** Installe `sentence-transformers`, encode 5 phrases (2 sur le même sujet, 3 sur des sujets différents), et construis la matrice de similarité cosinus entre toutes les paires. Vérifie que les 2 phrases du même sujet ont bien le score le plus élevé.

<details>
<summary>Solution</summary>

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

phrases = [
    "le chat dort sur le canapé",
    "un félin se repose tranquillement",
    "la bourse a chuté ce matin",
    "les marchés financiers sont en baisse",
    "je prépare une tarte aux pommes",
]
emb = model.encode(phrases)              # (5, 384)
sim = cosine_similarity(emb)             # matrice 5x5
print(sim.round(2))
```

Tu verras des blocs : les phrases (0,1) entre elles, et (2,3) entre elles, ont les cosinus les plus hauts.

</details>

#### ✅ Mini-checkpoint Jour 2

- ✓ Un **sentence embedding** encode une phrase entière en un vecteur dense
- ✓ Je sais l'obtenir avec `SentenceTransformer.encode(...)`
- ✓ Je vois pourquoi c'est mieux que la moyenne de Word2Vec
- ✓ La similarité se mesure **toujours** avec le cosinus

</details>

---

<details>
<summary>JOUR 3-4 : Moteur de recherche v0.4 — la recherche sémantique</summary>

#### 💻 Construire le moteur v0.4

C'est ton moteur v0.3, mais l'encodeur Word2Vec+moyenne est remplacé par un sentence embedding. **La logique de recherche est rigoureusement la même** (cosinus + tri).

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearchEngine:
    """Moteur v0.4 : recherche sémantique par sentence embeddings."""

    def __init__(self, documents, model_name="all-MiniLM-L6-v2"):
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        # On encode TOUS les documents UNE fois → c'est notre "index"
        self.doc_vectors = self.model.encode(documents, convert_to_numpy=True)

    def search(self, query, top_k=3):
        # 1. Transformer la requête en vecteur (même encodeur)
        q_vec = self.model.encode([query], convert_to_numpy=True)
        # 2. Cosinus entre la requête et chaque document
        scores = cosine_similarity(q_vec, self.doc_vectors)[0]
        # 3. Trier et renvoyer le top-k
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], float(scores[i])) for i in idx]


docs = [
    "Comment réinitialiser mon mot de passe ?",
    "Les horaires d'ouverture du magasin",
    "Procédure de remboursement d'une commande",
    "Mon colis est en retard, que faire ?",
]
engine = SemanticSearchEngine(docs)
for doc, score in engine.search("je n'arrive plus à me connecter à mon compte"):
    print(f"{score:.3f}  {doc}")
```

> 🔑 **Le point clé pédagogique** : la requête « je n'arrive plus à me connecter » ne partage **aucun mot** avec « réinitialiser mon mot de passe », pourtant le moteur les rapproche. **TF-IDF (Semaine 2) en serait incapable** (zéro mot commun → score 0). C'est ça, la recherche **sémantique** : on cherche par le **sens**, pas par les mots exacts.

#### 📖 Notion à connaître : le « vector store »

Encoder tous les documents à chaque requête serait absurde. En pratique on calcule les `doc_vectors` **une seule fois** et on les stocke dans un **vector store** (base de données vectorielle : FAISS, Chroma, Qdrant…). À la requête, on n'encode que la question et on cherche le top-k par cosinus — souvent via un index approché (ANN) pour aller vite sur des millions de documents.

> 💡 Tu n'as pas besoin d'installer une vraie base vectorielle aujourd'hui : ton `self.doc_vectors` en numpy **EST** déjà un vector store minimaliste. Un FAISS, c'est juste ça en (beaucoup) plus rapide et persistant.

#### 🎯 Exercice du jour

**Exercice :** Compare ton moteur v0.2 (TF-IDF) et v0.4 (sémantique) sur une requête qui **n'utilise aucun mot exact** des documents (synonymes, reformulation). Note les scores des deux. Que constates-tu ?

#### ✅ Mini-checkpoint Jour 3-4

- ✓ J'ai un moteur v0.4 qui cherche par le **sens**
- ✓ Je vois ce que la recherche sémantique fait que TF-IDF ne peut pas
- ✓ Je sais ce qu'est un **vector store** et que mon array numpy en est un

</details>

---

<details>
<summary>JOUR 5 : Le pont — « ce que tu viens de coder, c'est la base du RAG »</summary>

#### 📖 Le RAG, démystifié

Tu entends partout « RAG » (Retrieval-Augmented Generation) à propos des apps LLM. **Tu viens d'en coder les 2/3.** Le RAG, c'est 4 étapes :

```
   QUESTION de l'utilisateur
        │
        ▼  1. EMBED      → encoder la question         ← Jour 2 (encode)
        ▼  2. RETRIEVE   → top-k docs par cosinus      ← Jour 3-4 (ton moteur v0.4 !)
        ▼  3. AUGMENT    → coller ces docs dans le prompt
        ▼  4. GENERATE   → un LLM rédige la réponse à partir des docs
        │
        ▼  RÉPONSE sourcée
```

> 🔑 **Les étapes 1 et 2 sont LITTÉRALEMENT ton moteur v0.4.** Le RAG n'ajoute que : (3) mettre les documents trouvés dans le prompt d'un LLM, et (4) laisser le LLM rédiger. Tout le « retrieval » repose sur les **embeddings denses + cosinus** que tu maîtrises depuis la Semaine 1.

#### 💻 Le squelette d'un RAG (pseudo-code)

```python
def rag_answer(question, engine, llm):
    # 1+2. EMBED + RETRIEVE  → ton moteur v0.4
    passages = engine.search(question, top_k=3)
    contexte = "\n".join(doc for doc, score in passages)

    # 3. AUGMENT → construire le prompt
    prompt = f"""Réponds à la question en t'appuyant UNIQUEMENT sur le contexte.

Contexte :
{contexte}

Question : {question}
Réponse :"""

    # 4. GENERATE → appel au LLM (ex. API Claude)
    return llm.generate(prompt)
```

#### 🧭 Pourquoi c'est LE pont vers ton objectif (apps agentiques)

- Un **agent** = un LLM qui peut **agir** (chercher, appeler des outils, lire des documents). Sa capacité à « chercher dans une base de connaissances » **EST** un moteur de recherche sémantique comme ton v0.4.
- Le RAG est la brique de base de la plupart des apps LLM utiles (chatbots documentaires, assistants internes, support client). Tu en comprends maintenant **toute la moitié « retrieval »**, des fondations (cosinus) au sommet (vector store).

> 💡 Prochaine étape naturelle après cette roadmap : remplacer le `llm.generate(...)` factice par un vrai appel d'API (ex. Claude), puis donner au LLM **plusieurs outils** (pas juste la recherche) → c'est le passage du RAG à l'**agent**.

#### 🤔 Questions de réflexion

**Q1 :** Pourquoi le RAG est-il souvent préféré au « ré-entraînement du LLM » pour lui donner des connaissances métier ?

**Q2 :** Dans le RAG, à quel moment exact intervient la similarité cosinus que tu as apprise en Semaine 1 ?

**Q3 :** Quelle limite de TF-IDF la recherche sémantique résout-elle, et pourquoi est-ce crucial pour un chatbot ?

</details>

---

### ✅ Checkpoint Semaine 5

- ✓ Je sais passer d'un embedding de **mot** à un embedding de **document** (pooling, puis sentence embeddings)
- ✓ Je sais pourquoi la moyenne de Word2Vec ne suffit pas (ordre, contexte)
- ✓ Je sais utiliser `sentence-transformers` pour encoder des phrases
- ✓ J'ai un moteur de recherche **sémantique** (v0.4) et je vois ce qu'il fait de plus que TF-IDF
- ✓ Je comprends ce qu'est un **vector store**
- ✓ Je peux expliquer un pipeline **RAG** et situer où vivent les embeddings et le cosinus
- ✓ Je vois le pont concret entre cette roadmap et les **apps agentiques**

</details>

---

## 🎓 BILAN FINAL DE LA PHASE 1

### ✅ Ce que tu dois maîtriser maintenant

**Concepts théoriques**

- ✓ Bag of Words (comptage de mots)
- ✓ TF-IDF (valorisation des mots rares)
- ✓ Word embeddings (vecteurs denses sémantiques)
- ✓ Similarité cosinus (mesure de proximité)
- ✓ Hypothèse distributionnelle
- ✓ 🆕 *(Semaine 5)* Document embeddings : pooling vs **sentence embeddings**
- ✓ 🆕 *(Semaine 5)* Recherche sémantique, **vector store** et pipeline **RAG**

**Compétences pratiques**

- ✓ Coder BoW de A à Z
- ✓ Coder TF-IDF de A à Z
- ✓ Utiliser sklearn pour le NLP
- ✓ Utiliser Word2Vec (gensim)
- ✓ Entraîner un modèle Word2Vec
- ✓ Construire un moteur de recherche
- ✓ 🆕 *(Semaine 5)* Encoder des phrases avec `sentence-transformers`
- ✓ 🆕 *(Semaine 5)* Assembler le squelette d'un système RAG (embed → retrieve → augment → generate)

**Projets réalisés**

- ✓ Moteur v0.1 (BoW)
- ✓ Moteur v0.2 (TF-IDF)
- ✓ Moteur v0.3 (Word2Vec)
- ✓ Moteur hybride (combinaison)
- ✓ 🆕 *(Semaine 5)* Moteur v0.4 (recherche sémantique par sentence embeddings)

---

### 📊 Auto-évaluation

Note-toi de 1 à 5 sur chaque point :

| Compétence | Note /5 | Commentaire |
|------------|---------|-------------|
| Je comprends BoW | | |
| Je comprends TF-IDF | | |
| Je comprends Word2Vec | | |
| 🆕 Je comprends les sentence embeddings (Semaine 5) | | |
| 🆕 Je comprends le pipeline RAG (Semaine 5) | | |
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

### 🆕 Phase 5 : Apps agentiques *(prolonge la Semaine 5)*

- Brancher un vrai LLM (ex. API Claude) sur ton moteur v0.4 → RAG complet
- Vector store persistant (FAISS/Chroma) sur un vrai corpus
- Donner plusieurs **outils** au LLM (pas que la recherche) → passage du RAG à l'**agent**

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
- 🆕 **sentence-transformers** : sentence/document embeddings (Semaine 5)
- 🆕 **FAISS / Chroma / Qdrant** : vector stores pour la recherche sémantique (Semaine 5)

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
