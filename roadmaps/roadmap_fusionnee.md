# 🗺️ ROADMAP FUSIONNÉE — Deux racines qui convergent vers les LLM

> **Idée directrice.** Deux traditions distinctes — la **recherche d'information / représentation** (BoW, TF-IDF, embeddings) et la **probabilité / modélisation du langage** (Naive Bayes, n-grammes, théorie de l'information) — partagent le même socle et **finissent par fusionner** dans le Transformer. La voie représentation mène au *retrieval* du RAG ; la voie langage mène au *générateur* (le LLM lui-même). On les fait avancer en parallèle au lieu de l'une après l'autre.
>
> **Le principe à garder en tête tout du long :** la question ne change presque jamais — c'est toujours estimer **P(quelque chose | contexte)**. Seul l'outil monte en puissance : *compter des proportions* → *apprendre des poids* → *attention*.

**Objectif final de la Phase 1 :** transformer du texte en vecteurs, construire un moteur de recherche qui s'améliore (v0.1 → v0.4), **et** comprendre « prédire le mot suivant » — l'objectif d'entraînement d'un LLM.

**Durée réaliste :** 8-10 semaines (la voie langage ~double le contenu de la roadmap IR seule). Flexible selon ton rythme.

---

## 🎨 Code couleur

- 🟩 **[représentation / IR]** — comptage, vecteurs, recherche
- 🟪 **[langage / proba]** — proportions, prédiction, log-probas
- ⬜ **[tronc commun / fusion]** — briques partagées et points de jonction

---

## 🧭 Schéma d'ensemble

```
        Tronc commun — le texte devient des vecteurs (S1)
        one-hot, BoW, cosinus → moteur v0.1
                          │
        ┌─────────────────┴──────────────────┐
        ▼                                     ▼
🟩 VOIE REPRÉSENTATION / IR          🟪 VOIE PROBABILITÉ / LANGAGE
        │                                     │
   TF-IDF (S2)                          Naive Bayes revisité (S2)
   pondérer mots rares → v0.2           proba, log, Bayes, lissage
        │                                     │
        │                              n-grammes (S3)
        │                              prédire le mot suivant
        │                                     │
        │                              Théorie de l'information (S3)
        │                              entropie, surprise, cross-entropy
        │                                     │
        └──────────────┬──────────────────────┘
                       ▼
        Apprendre les poids au lieu de les compter (S4)
        régression logistique → couche d'embedding entraînable
        (LA bascule comptage → apprentissage ; gradient introduit UNE fois)
                       │
        ┌──────────────┴──────────────────────┐
        ▼                                     ▼
   Word2Vec (S5)                       (la prédiction refait surface :
   vecteurs denses du sens → v0.3       skip-gram = prédire le contexte)
        │
   Sentence embeddings + RAG (S6)
   recherche sémantique → v0.4
                       │
                       ▼
        Phase 2+ — FUSION : réseaux de neurones → RNN → attention → Transformer
        retrieval (voie repr.) + génération (voie langage) = RAG / agent
        → bascule vers roadmap_ml_explications.md
```

---

## ✅ Acquis (déjà fait — point de départ)

- 🟩 **BoW** codé à la main (`bow/bow_from_scratch.py`), moteur de recherche v0.1 (`bow/search_engine_v01.py`)
- 🟩 **Similarité cosinus** comprise et codée
- 🟪 **Intuitions probabilistes** (digression Naive Bayes) : proportions, multiplication en cascade (filtrage), conditionnement `P(B|A)`, théorème de Bayes, passage au log
- 🟪 Notebook **`naive_bayes_spam.ipynb`**

> 🔜 **Avant d'attaquer la Semaine 3, finis TF-IDF (Semaine 2).** Ferme le fil, pose le moteur v0.2, puis pars sur les n-grammes la tête libre.

---

<details>
<summary><strong>📅 SEMAINE 1 — ⬜ Tronc commun : le texte devient des vecteurs</strong></summary>

### 🎯 Objectif
Une seule opération fonde tout : transformer des symboles discrets en nombres manipulables.

### 📖 Concepts clés
- **one-hot encoding** = la brique atomique : un mot → un vecteur de zéros avec un seul 1.
- **BoW = somme de one-hots** ; **TF-IDF = version pondérée** (Semaine 2).
- **tokeniser ≠ vectoriser** : le tokeniser produit des *indices* (entiers), le vectoriser des *vecteurs*.
- Tous les one-hots sont **orthogonaux** → cosinus = 0 entre n'importe quels deux mots → **aucune notion de sens**. C'est la motivation de tout ce qui suit.
- **Similarité cosinus** comme mesure de proximité.

### 🌉 Pont
Tout part de la même opération `symbole → nombre`. BoW, TF-IDF, embeddings n'en sont que des raffinements.

### 💻 Artefact
Moteur de recherche **v0.1** (déjà fait) + ajouter `one_hot.py` pour nommer explicitement la brique atomique.

### ✅ Checkpoint
- [ ] Je sais écrire le one-hot d'un mot et expliquer son orthogonalité
- [ ] Je sais que BoW = somme de one-hots
- [ ] Je distingue tokeniser (→ indices) et vectoriser (→ vecteurs)

> 📎 Détail complet et squelettes : voir `vectorisation_roadmap_amelioree.md`, Semaine 1 + Semaine 2.5 (Jour 1).

</details>

---

<details open>
<summary><strong>📅 SEMAINE 2 — Deux pondérations, deux questions</strong></summary>

### 🎯 Objectif
Mettre côte à côte les deux racines à la même hauteur : pondérer (IR) vs classer (proba).

---

#### 🟩 [représentation] TF-IDF — valoriser les mots rares

- **Problème de BoW** : « le » compte beaucoup mais n'informe pas.
- **TF** = fréquence du terme dans le document ; **IDF** = `log(N / nb_docs_contenant_le_mot)` → écrase les mots partout, valorise les rares.
- **TF-IDF = TF × IDF**. Moteur **v0.2**.

> 📎 Squelette complet : `vectorisation_roadmap_amelioree.md`, Semaine 2.

---

#### 🟪 [langage] Naive Bayes revisité proprement

Tu l'as fait en digression — ici on le **consolide** comme membre de la voie langage.

- `P(classe | doc) ∝ P(classe) × P(doc | classe)` (Bayes).
- Hypothèse **naïve** : les mots sont indépendants *sachant la classe* → `P(doc|classe) ≈ ∏ P(mot|classe)`.
- Produit de petites probas → **somme de log** (stabilité numérique).
- **Lissage de Laplace** : un seul `P(mot|classe)=0` annule tout le produit → on ajoute `+1`.

**Exercice de consolidation** (papier) : redérive `P(spam|mots) = P(spam)·∏P(mot|spam) / P(mots)` à partir de `P(A et B) = P(A)·P(B|A)`, et explique pourquoi le dénominateur `P(mots)` disparaît dans la comparaison spam vs ham.

---

### 🌉 Pont (le cœur de la semaine)
**L'additivité est la brique partagée.** Le « naïf » de Naive Bayes (indépendance des mots) **est** le « sac » de bag-of-words (oubli de l'ordre). Cette hypothèse est précisément ce qui te permet de décomposer une quantité au niveau du document en **somme de contributions par terme** :
- Naive Bayes : `log P(classe|doc) = log P(classe) + Σ log P(mot|classe)`
- TF-IDF : `score(doc) = Σ poids(terme)`

Même geste structurel. **Différence clé** : Naive Bayes *multiplie* des probas (d'où le danger du zéro → lissage indispensable) ; TF-IDF *additionne* des poids (un terme à zéro contribue zéro, sans tout annuler). C'est pour ça que TF-IDF a besoin que tu *comprennes* le principe d'additivité sans avoir à le manipuler comme Naive Bayes.

### ✅ Checkpoint Semaine 2
- [ ] Moteur v0.2 (TF-IDF) fonctionnel, comparé à v0.1
- [ ] Je sais réénoncer Naive Bayes comme produit → somme de log
- [ ] Je sais expliquer pourquoi le lissage de Laplace est vital en multiplicatif mais pas en additif
- [ ] Je vois que « naïf » = « sac » = hypothèse d'indépendance

</details>

---

<details>
<summary><strong>📅 SEMAINE 3 — 🟪 Prédire le mot suivant, et pourquoi le log</strong></summary>

### 🎯 Objectif
Coder ton premier **modèle de langue** par comptage, et comprendre enfin *pourquoi* on travaille en log. C'est la récolte de tout ton travail sur les probas.

---

<details open>
<summary>JOUR 1-3 : 🟪 n-grammes à la main</summary>

#### 📖 Concepts
Un **modèle de langue** estime `P(mot | contexte)`. Le n-gramme fait l'hypothèse de Markov : le contexte se limite aux `n−1` mots précédents.

```
unigramme : P(mot)                      ← pas de contexte
bigramme  : P(mot | mot_précédent)      ← le bigram de Karpathy (n=2)
trigramme : P(mot | 2 mots précédents)
n-gramme  : P(mot | n−1 mots précédents)
```

> 🔑 **Lien avec ce que tu connais.** C'est *exactement* l'outillage Naive Bayes : compter, transformer en proportion, lisser (Laplace), passer en log. Tu ne changes que la question : « quel mot après ce contexte ? » Or **c'est littéralement ce que fait un LLM**.
>
> 🔑 **Lien avec Karpathy.** Ton bigramme par *comptage* = la première moitié de son cours. Plus tard (Semaine 4), on refera le même bigramme en *l'apprenant* par gradient — et on verra que les deux convergent. Garde ce fil en tête.

#### ✏️ Exercice papier
Corpus : `"le chat dort"`, `"le chat mange"`, `"le chien dort"`.
Calcule `P(dort | chat)` et `P(mange | chat)` par comptage. Puis applique Laplace (`+1`, vocabulaire de taille V) et recalcule.

<details><summary>Solution</summary>

Sans lissage : « chat » est suivi de « dort » 1 fois, « mange » 1 fois → `P(dort|chat) = P(mange|chat) = 1/2`.
Avec Laplace (`+1`, V mots distincts) : `P(dort|chat) = (1+1)/(2+V)`. Le lissage donne une proba non nulle aux mots jamais vus après « chat ».
</details>

#### 💻 Squelette : n-grammes from scratch (comptage + Laplace + génération)

```python
# ngram_from_scratch.py
import math
from collections import Counter, defaultdict
import random

def tokenize(text):
    return text.lower().split()

def build_ngram_counts(corpus, n):
    """
    Compte les n-grammes.
    Returns:
        context_counts: Counter {contexte (tuple de n-1 mots): total}
        ngram_counts:   dict {contexte: Counter {mot_suivant: compte}}
        vocab:          set des mots
    """
    context_counts = Counter()
    ngram_counts = defaultdict(Counter)
    vocab = set()
    for sentence in corpus:
        tokens = ["<s>"] * (n - 1) + tokenize(sentence) + ["</s>"]
        vocab.update(tokens)
        # TODO :
        # 1. faire glisser une fenêtre de taille n sur tokens
        # 2. contexte = les n-1 premiers, mot = le dernier
        # 3. incrémenter context_counts[contexte] et ngram_counts[contexte][mot]
    return context_counts, ngram_counts, vocab

def proba(mot, contexte, context_counts, ngram_counts, vocab, alpha=1.0):
    """P(mot | contexte) avec lissage de Laplace (alpha)."""
    V = len(vocab)
    # TODO : (compte(contexte, mot) + alpha) / (compte(contexte) + alpha * V)
    pass

def log_proba_phrase(sentence, n, context_counts, ngram_counts, vocab, alpha=1.0):
    """Somme des log P(mot|contexte) sur la phrase — la log-vraisemblance."""
    tokens = ["<s>"] * (n - 1) + tokenize(sentence) + ["</s>"]
    total = 0.0
    # TODO : pour chaque position, contexte = n-1 précédents ;
    #        total += math.log(proba(mot, contexte, ...))
    return total

def generate(n, context_counts, ngram_counts, vocab, max_len=20):
    """Génère une phrase en échantillonnant mot à mot."""
    contexte = tuple(["<s>"] * (n - 1))
    out = []
    for _ in range(max_len):
        # TODO :
        # 1. récupérer la distribution des mots suivants pour `contexte`
        # 2. échantillonner un mot (random.choices avec les poids)
        # 3. si "</s>", arrêter ; sinon append et faire glisser le contexte
        pass
    return " ".join(out)

if __name__ == "__main__":
    corpus = [
        "le chat dort sur le canapé",
        "le chat mange la souris",
        "le chien dort dans le jardin",
        "le chien aboie fort",
    ]
    n = 2  # commence par le bigramme, puis essaie n=3
    cc, nc, vocab = build_ngram_counts(corpus, n)
    print("P(dort|le chat) =", proba("dort", ("chat",), cc, nc, vocab))
    print("Phrase générée :", generate(n, cc, nc, vocab))
```

#### 🔬 Expériences
1. Passe de `n=2` à `n=3` : la génération devient plus cohérente mais le modèle « copie » davantage le corpus (sur-apprentissage). Pourquoi ?
2. Enlève le lissage (`alpha=0`) et génère une phrase avec un contexte rare → observe le crash / les zéros.

</details>

---

<details open>
<summary>JOUR 4-5 : 🟪 Théorie de l'information (léger mais décisif)</summary>

#### 📖 L'idée, en 3 notions
- **Surprise** d'un événement : `−log(p)`. Plus un mot est improbable, plus le voir est « surprenant ». (C'est *pourquoi* on travaille en log : log transforme un produit qui plonge vers 0 en une somme, et la quantité naturelle « surprise » est déjà un log.)
- **Entropie** : la surprise *moyenne* d'une distribution.
- **Cross-entropy** : la surprise moyenne de **tes prédictions face à la réalité**. C'est exactement `−(1/N) Σ log P(mot réel | contexte)` — la moyenne de la `log_proba_phrase` que tu viens de coder, au signe près.

> 🔑 **La récolte.** La cross-entropy d'un n-gramme **est, en miniature, la fonction de coût d'un LLM.** Entraîner un LLM = minimiser la surprise moyenne sur le texte. Tout ton travail sur les log-probas prend ici son sens rétrospectif.
>
> La **perplexité** = `exp(cross-entropy)` : « le modèle hésite en moyenne entre combien de mots ? ». C'est la métrique standard des modèles de langue.

#### ✏️ Exercice
À partir de ta fonction `log_proba_phrase`, code `cross_entropy(corpus_test, ...)` et `perplexite(...)`. Compare la perplexité d'un bigramme et d'un trigramme sur une phrase de test. Lequel est le plus « confiant » ? Au prix de quoi ?

#### ✅ Checkpoint Semaine 3
- [ ] J'ai un modèle n-grammes qui compte, lisse, calcule des log-probas et génère du texte
- [ ] Je sais que bigramme = n-gramme avec n=2 (le bigram de Karpathy)
- [ ] Je sais expliquer `surprise = −log(p)` et pourquoi on additionne des log-probas
- [ ] Je sais que cross-entropy = fonction de coût d'un LLM, et calculer une perplexité

</details>

</details>

---

<details>
<summary><strong>📅 SEMAINE 4 — ⬜ Apprendre les poids au lieu de les compter</strong></summary>

### 🎯 Objectif
**LE pivot conceptuel de toute la roadmap.** Jusqu'ici tes « poids » sont des comptages figés (tu observes, tu divises). Ici, pour la première fois, le modèle **ajuste** ses poids pour coller aux données : c'est la **descente de gradient**, le cœur de tout l'apprentissage.

> ⚠️ **Ordre important (gradient introduit UNE seule fois).** On fait la **régression logistique d'abord** (le cas le plus simple : un seul neurone), *puis* la **couche d'embedding entraînable** (qui réutilise exactement le même mécanisme de gradient). Ne réintroduis pas le gradient deux fois.

---

<details open>
<summary>JOUR 1-3 : 🟪→🟩 Régression logistique à la main (un neurone)</summary>

#### 📖 Concepts
- Même **entrée** que Naive Bayes (un sac de mots), même **sortie** (une classe) — mais les poids sont **appris**, pas comptés.
- **sigmoïde** `σ(z) = 1/(1+e^−z)` (2 classes) / **softmax** (multi-classes) : transforme des scores en proportions qui somment à 1.
- **Fonction de coût** : la **cross-entropy** (celle de la Semaine 3 !).
- **Descente de gradient** : `poids ← poids − lr × gradient`. On répète jusqu'à ce que le coût baisse.

> 🔑 **C'est le premier vrai « réseau de neurones » déguisé** — un neurone unique. Et `softmax` est *exactement* « transformer des scores en proportions », ton intuition de la digression Naive Bayes.

#### 💻 Squelette : régression logistique from scratch (numpy)

```python
# logistic_regression.py
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def train(X, y, lr=0.1, epochs=200):
    """
    X : (n_exemples, n_features) — sacs de mots
    y : (n_exemples,) — 0 ou 1
    """
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for epoch in range(epochs):
        z = X @ w + b
        p = sigmoid(z)                 # proba prédite
        # cross-entropy (pour suivre la baisse du coût)
        loss = -np.mean(y*np.log(p+1e-9) + (1-y)*np.log(1-p+1e-9))
        # gradients (dérivée de la cross-entropy + sigmoïde)
        dz = p - y                     # ← même forme que (y_pred - y_vrai)
        # TODO :
        # grad_w = X.T @ dz / n
        # grad_b = dz.mean()
        # w -= lr * grad_w ; b -= lr * grad_b
        if epoch % 50 == 0:
            print(f"epoch {epoch:3d}  loss={loss:.3f}")
    return w, b
```

> 🌉 **Raccord avec Karpathy.** Souviens-toi du bigramme par comptage (Semaine 3). Karpathy le refait comme **un seul neurone entraîné par gradient + softmax** sur la même tâche `P(mot suivant | mot précédent)` — et montre qu'il **converge vers les mêmes proportions** que le comptage. C'est *la* démonstration que « compter » et « apprendre » sont deux façons d'atteindre la même `P(·|contexte)`. → suite dans `roadmap_ml_explications.md`.

</details>

---

<details open>
<summary>JOUR 4-5 : 🟩 Couche d'embedding entraînable</summary>

#### 📖 Concepts
- Une **couche d'embedding** est une matrice `W` de taille `(vocab × dim)`. Chaque **ligne** = le vecteur dense d'un mot.
- `one-hot × W = la ligne correspondante` → en pratique on fait juste `W[indice]` (le « lookup »).
- Différence avec one-hot/TF-IDF : les valeurs sont **apprises** par gradient (le mécanisme que tu viens de voir), pas fixées par des règles.
- On entraîne `W` pour qu'un mot serve à **prédire ses voisins** (hypothèse distributionnelle) → c'est déjà du **skip-gram**.

#### 💻 Squelette : skip-gram à la main (une couche)
> 📎 Le squelette numpy complet existe déjà : `vectorisation_roadmap_amelioree.md`, Semaine 2.5, Jour 3-4 (`train_embeddings.py`). On garde `W_emb` (les embeddings), on jette `W_out`.

### 🌉 Pont
**Les deux voies se rejoignent ici une première fois.** La bascule « compter » (Bayes, TF-IDF, n-grammes) → « apprendre » (gradient) vaut pour la classification (régression logistique) *et* pour la représentation (embedding). Même mécanisme, deux usages.

### ✅ Checkpoint Semaine 4
- [ ] J'ai entraîné un classifieur (régression logistique) par descente de gradient
- [ ] Je relie softmax/sigmoïde + cross-entropy + gradient au résultat
- [ ] J'ai entraîné une couche d'embedding à la main (`W_emb` gardé, `W_out` jeté)
- [ ] Je comprends que le bigramme par comptage et le bigramme appris convergent

</details>

</details>

---

<details>
<summary><strong>📅 SEMAINE 5 — 🟩 Word2Vec : les vecteurs denses du sens</strong></summary>

### 🎯 Objectif
Des vecteurs denses (100-300 dim) où des mots de sens proche sont proches dans l'espace.

### 📖 Concepts
- **Hypothèse distributionnelle** (Firth) : « un mot est caractérisé par la compagnie qu'il tient » — la racine commune de l'IR *et* des embeddings.
- **CBOW** (prédire le mot central depuis le contexte) vs **skip-gram** (prédire le contexte depuis le mot central).
- `gensim.Word2Vec(...)` = ce que tu as codé à la main en Semaine 4, en (beaucoup) plus optimisé.
- **Analogies** : `roi − homme + femme ≈ reine` (les relations sémantiques = translations dans l'espace).
- Moteur **v0.3** (document = moyenne des vecteurs de ses mots).

### 🌉 Pont
**La voie langage refait surface au cœur de la voie représentation.** L'objectif de Word2Vec (skip-gram = prédire le contexte) est lui-même une **tâche de prédiction**. Les embeddings ne sont qu'un sous-produit d'un modèle de langue minuscule.

> ⚠️ **Nuance technique** : le cosinus va en général de **−1 à 1** (et non 0 à 1 comme avec des vecteurs de comptages toujours positifs). Avec des embeddings denses, des composantes négatives existent.

### ✅ Checkpoint
- [ ] Moteur v0.3 fonctionnel, synonymes capturés
- [ ] Je relie mon code Semaine 4 à `gensim.Word2Vec`
- [ ] Je sais que skip-gram = une tâche de prédiction (voie langage)

> 📎 Détail complet, gensim, visualisations, analogies : `vectorisation_roadmap_amelioree.md`, Semaine 3.

</details>

---

<details>
<summary><strong>📅 SEMAINE 6 — 🟩 Du mot au document : recherche sémantique et RAG</strong></summary>

### 🎯 Objectif
Obtenir **un seul vecteur par document**, puis assembler un squelette **RAG**.

### 📖 Concepts
- **Pooling** (moyenne des vecteurs de mots) et ses limites : commutatif → perd l'**ordre** (`chat mange souris` = `souris mange chat`) et le **contexte** (mots polysémiques).
- **Sentence embeddings** (`sentence-transformers` / SBERT) : un modèle entraîné qui encode une phrase entière en tenant compte de l'ordre et du contexte.
- Moteur **v0.4** (recherche sémantique) + **vector store** (FAISS/Chroma/Qdrant ; ton array numpy en est un, minimaliste).
- **RAG** = `embed → retrieve → augment → generate`. Les étapes 1-2 **sont** ton moteur v0.4.

### 💸 Dette assumée
`sentence-transformers` **est déjà un Transformer** utilisé en boîte noire. Tu vois le résultat avant le mécanisme — dette remboursée en Phase 2+.

### ✅ Checkpoint
- [ ] Moteur v0.4 sémantique, comparé à TF-IDF sur une requête sans mot commun
- [ ] Je sais ce qu'est un vector store
- [ ] Je sais situer embeddings et cosinus dans un pipeline RAG

> 📎 Détail complet : `vectorisation_roadmap_amelioree.md`, Semaine 5.

</details>

---

## 🚀 PHASE 2 ET AU-DELÀ — La fusion complète

> 🔗 **Raccord explicite.** À partir d'ici, on bascule sur l'autre roadmap du repo : **`roadmap_ml_explications.md`** (séquence Karpathy : Bigram → Micrograd → n-gram/MLP → attention → Transformer). Cette roadmap-ci en est le **socle** ; celle-là en est la **suite neuronale**.

| Étape | Ce que ça ajoute | Relie à |
|---|---|---|
| **Réseaux de neurones + backprop** (Micrograd) | Généralise le neurone unique de la Semaine 4 à plusieurs couches | S4 |
| **RNN / LSTM** | La mémoire de séquence : traiter les mots un par un en gardant un état | voie langage |
| **Attention → Transformer** | Chaque mot regarde directement n'importe quel autre mot (résout le goulot des RNN) | fusion |
| **RAG complet → agent** | retrieval (voie repr.) + génération (voie langage) ; puis plusieurs outils | les deux voies ne font plus qu'un |

> 🔑 Quand tu arriveras au Transformer, tu auras déjà l'intuition du **quoi** (modéliser `P(mot suivant | contexte)`, ta brique n-grammes) ; il ne restera que le **comment** (le faire avec des poids appris et de l'attention, au lieu de comptages).

---

## 📚 Ressources qui appuient cette roadmap

1. **Jurafsky & Martin — *Speech and Language Processing* (3ᵉ éd., brouillon gratuit)** — la référence n°1, le découpage colle aux semaines : n-grammes, Naive Bayes, régression logistique, sémantique vectorielle, réseaux de neurones, Transformers. → `web.stanford.edu/~jurafsky/slp3`
2. **David MacKay — *Information Theory, Inference, and Learning Algorithms* (gratuit)** — pour la Semaine 3 (entropie, surprise, vraisemblance). Premiers chapitres suffisants. → `inference.org.uk`
3. **Simon Prince — *Understanding Deep Learning* (PDF gratuit)** — pour la Phase 2+ (réseaux → attention → Transformers). → `udlbook.github.io/udlbook`
4. *(bonus, code-first)* **Dive into Deep Learning** → `d2l.ai`
5. *(bonus, historique)* **Manning & Schütze — *Foundations of Statistical NLP*** (payant, daté, rigoureux).

---

## 💡 Conseils

- **Pratique > théorie** : code chaque brique, ne te contente pas de lire.
- **Compare toujours** à la méthode précédente (v0.1 vs v0.2 vs …).
- **Traque les briques partagées** entre les deux voies : c'est ce qui rendra le Transformer non magique.
- **Ne reste pas bloqué** : après 2h sur un point, passe à la suite et reviens l'esprit frais.
- **Enseigne** ce que tu apprends (le meilleur test de compréhension).
