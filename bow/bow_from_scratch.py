import math

    # Pour lancer le script : python bow/bow_from_scratch.py

def create_vocabulary(documents):
    all_words = []

    for doc in documents:
        doc = doc.lower()
        words = doc.split()
        all_words.extend(words)
    
    unique_words = set(all_words)
    vocabulary = sorted(unique_words)

    return vocabulary


def document_to_vector(document, vocabulary):
    # Créer un vecteur de zéros 
    vector = [0] * len(vocabulary)
    document = document.lower()
    words = document.split()

    for word in words:
        if word in vocabulary:
            index = vocabulary.index(word)
            vector[index] += 1

    return vector

def cosine_similarity(vec1, vec2):
    # Produit scalaire
    dot_product = 0
    for i in range(len(vec1)):
        dot_product += vec1[i] * vec2[i]
    
    norm1 = 0
    for val in vec1:
        norm1 += val ** 2  # val au carré
    norm1 = math.sqrt(norm1)  # Racine carrée
    
    norm2 = 0
    for val in vec2:
        norm2 += val ** 2
    norm2 = math.sqrt(norm2)

    if norm1 == 0 or norm2 == 0:
        return 0.0  # Éviter division par zéro

    similarity = dot_product / (norm1 * norm2)

    return similarity  # Pour l'instant


documents = [
    "le chat dort sur le tapis",
    "le chien aboie dans le jardin",
    "le chat mignon joue"
]

vocab = create_vocabulary(documents)
print("Vocabulaire:", vocab)
print()

vectors = []

for i, doc in enumerate(documents):
    vec = document_to_vector(doc, vocab)
    vectors.append(vec)
    print(f"Doc {i+1}: {doc}")
    print(f"Vecteur: {vec}")
    print()


sim = cosine_similarity(vectors[0], vectors[2])
print(f"Similarité doc1 vs doc3: {sim:.3f}")
