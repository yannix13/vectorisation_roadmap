import math

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


class SimpleSearchEngine:
    def __init__(self, documents):
        """
        Initialise le moteur avec une collection de documents
        
        Args:
            documents: liste de strings
        """
        print("Initialisation du moteur de recherche...")
        
        # Sauvegarder les documents
        self.documents = documents
        print(f"✓ {len(documents)} documents chargés")

        # Créer le vocabulaire
        self.vocabulary = create_vocabulary(documents)
        print(f"✓ Vocabulaire créé : {len(self.vocabulary)} mots")

        # Vectoriser tous les documents
        self.doc_vectors = []
        for doc in documents:
            vec = document_to_vector(doc, self.vocabulary)
            self.doc_vectors.append(vec)
        
        print(f"✓ {len(self.doc_vectors)} documents vectorisés")
        print("\nMoteur prêt ! 🚀")


    def search(self, query, top_k=3):
        """
        Recherche les documents les plus pertinents
        
        Args:
            query: string (requête de l'utilisateur)
            top_k: nombre de résultats à retourner
        
        Returns:
            liste de tuples (index_document, score_similarité)
        """
        print(f"\n🔍 Recherche : '{query}'")
        
        # 1. Transformer la requête en vecteur
        query_vector = document_to_vector(query, self.vocabulary)
        print(f"✓ Requête vectorisée")
        
        # 2. Calculer similarité avec chaque document
        similarities = []
        for i, doc_vec in enumerate(self.doc_vectors):
            sim = cosine_similarity(query_vector, doc_vec)
            similarities.append((i, sim))
            print(f"  Doc {i}: similarité = {sim:.3f}")
        
        # 3. Trier par score décroissant
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 4. Retourner top_k résultats
        return similarities[:top_k]
    

    def display_results(self, results):
        """
        Affiche les résultats de manière lisible
        """
        print("\n" + "="*50)
        print("RÉSULTATS DE RECHERCHE")
        print("="*50)
        
        for rank, (doc_idx, score) in enumerate(results, 1):
            print(f"\n{rank}. Score: {score:.3f}")
            print(f"   {self.documents[doc_idx]}")
        
        print("="*50)

if __name__ == "__main__":
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

    # Boucle de recherche interactive
    print("\n" + "="*50)
    print("MOTEUR DE RECHERCHE v0.1")
    print("="*50)
    print("Tapez votre recherche ou 'quit' pour quitter\n")
    
    while True:
        query = input("🔍 Recherche : ")
        
        if query.lower() == 'quit':
            print("\nAu revoir ! 👋")
            break
        
        if query.strip() == "":
            continue
        
        results = engine.search(query, top_k=3)
        engine.display_results(results)

