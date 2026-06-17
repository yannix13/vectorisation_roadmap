# Comprendre la Roadmap ML : Modèles vs Outils

> Documentation issue d'une conversation pédagogique sur les fondamentaux du Machine Learning et du NLP.

---

## 1. La Roadmap originale et ses problèmes

### Roadmap initiale

| Chapitre | Sujet |
|---|---|
| 01 | Bigram Language Model |
| 02 | Micrograd |
| 03 | N-gram model (MLP, matmul, gelu) |
| 04 | Attention (softmax, positional encoder) |
| 05 | Transformer (residual, layernorm, GPT-2) |
| 06 | Tokenization (minBPE, byte pair encoding) |
| 07 | Optimization (initialization, AdamW) |

### Problèmes identifiés

- **Tokenization (chap. 6) arrive trop tard** : c'est un prérequis pour comprendre comment le texte devient des nombres, utilisé implicitement dès le chapitre 1.
- **Micrograd (chap. 2) est mal placé** : on passe du bigram à la théorie du backprop, puis on revient aux n-grams — va-et-vient conceptuel perturbant.
- **Optimization (chap. 7) trop tard** : on utilise des optimizers dès les chapitres 3-4-5 sans comprendre ce qu'ils font.

### Ordre recommandé

1. **Tokenization** — Comment le texte devient des nombres
2. **Bigram** — Premier modèle simple de langage
3. **Micrograd** — Fondations du backprop
4. **N-gram / MLP** — Premiers réseaux de neurones
5. **Optimization** — Comprendre AdamW avant les gros modèles
6. **Attention** — Mécanisme clé
7. **Transformer** — Architecture complète

---

## 2. Deux catégories fondamentales à distinguer

La confusion principale de la roadmap vient du fait qu'elle mélange deux types d'éléments très différents.

### Les Modèles (architectures)

Ce sont des réponses à la question : **"Comment je structure mon modèle ?"**

- **Bigram** → architecture de modèle
- **N-gram** → architecture de modèle
- **Transformer** → architecture de modèle

### Les Outils (moteurs d'entraînement)

Ce sont des réponses à la question : **"Comment j'entraîne mon modèle ?"**

- **Micrograd** → moteur d'entraînement (pédagogique, from scratch)
- **PyTorch** → moteur d'entraînement (production)
- **TensorFlow** → moteur d'entraînement (production)

> Ces deux axes sont **perpendiculaires**, pas séquentiels. On peut entraîner n'importe quel modèle avec n'importe quel outil :
> - Bigram + PyTorch ✓
> - Transformer + Micrograd ✓
> - N-gram + TensorFlow ✓

---

## 3. Les modèles de langage : progression naturelle

### Bigram

Le modèle le plus simple. Il prédit le token suivant en se basant **uniquement sur le token précédent**.

```python
# Table de probabilités (token -> token suivant)
transitions = {}

# Entraînement : compter les transitions
for i in range(len(sequence) - 1):
    current = sequence[i]
    next_token = sequence[i + 1]
    if current not in transitions:
        transitions[current] = {}
    transitions[current][next_token] = transitions[current].get(next_token, 0) + 1

# Prédiction
predict("chat") → {"dort": 0.7, "mange": 0.3}
```

**Points clés :**
- Environ 50-100 lignes de Python
- Essentiellement un dictionnaire de probabilités
- Permet déjà de faire de la **prédiction** et de la **génération de texte**
- Limite : aucune mémoire au-delà d'un token

**Exemple de génération :**
```python
token = "le"
texte = ["le"]

for _ in range(10):
    token = sample(predict(token))
    texte.append(token)

# "le chat dort sur le toit du chat mange le"
```

### N-gram

Même principe que le Bigram, mais il prend **plusieurs tokens de contexte** en compte.

```
Bigram  : "canapé" → prédit le suivant
Trigram : "dort sur" → prédit le suivant
N-gram  : N tokens → prédit le suivant
```

### Transformer

Architecture moderne qui prend **tout le contexte disponible** en compte grâce au mécanisme d'attention.

---

### Progression naturelle

```
Bigram      → prédit, mais sans contexte
N-gram      → prédit avec quelques tokens de contexte
Transformer → prédit avec tout le contexte
```

---

## 4. Micrograd en détail

### Ce que c'est

Micrograd est un mini-framework de deep learning créé par **Andrej Karpathy**. Il implémente le **backpropagation from scratch** pour montrer comment fonctionne la dérivation automatique (autograd) en construisant un graphe computationnel.

```python
class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __add__(self, other):
        # Construire le nœud, implémenter backward
        ...

    def __mul__(self, other):
        # Pareil
        ...

    def backward(self):
        # Topological sort + backprop récursif
        ...
```

Environ **300+ lignes** de Python.

### Pourquoi c'est important

Sans Micrograd, `.backward()` de PyTorch ressemble à de la magie noire. Après Micrograd, on comprend exactement ce qui se passe : calcul des gradients via la règle de la chaîne, mis à jour des poids.

### L'analogie

- Utiliser PyTorch = **conduire une voiture**
- Construire Micrograd = **démonter le moteur pour comprendre comment il fonctionne**

---

## 5. Micrograd vs PyTorch

| Caractéristique | Micrograd | PyTorch |
|---|---|---|
| Lignes de code | ~300 | Millions |
| GPU | ❌ | ✓ |
| Performance | Très lente | Très optimisée |
| Scalabilité | Scalaires uniquement | Tenseurs |
| Usage en production | ❌ | ✓ |
| Objectif | Pédagogique | Production |

### La limite fondamentale : scalaires vs tenseurs

**Micrograd** ne gère que des scalaires (nombres simples) :

```python
a = Value(2.0)
b = Value(3.0)
c = a * b  # ça marche, mais un nombre à la fois
```

**PyTorch** gère des tenseurs (matrices entières) :

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
c = a @ b  # multiplication matricielle ultra-optimisée
```

En vrai ML, on manipule des **millions de paramètres simultanément**. Micrograd serait beaucoup trop lent.

### Conclusion sur Micrograd

| | |
|---|---|
| ✅ | Parfait pour **comprendre** le backprop |
| ✅ | Parfait pour **apprendre** |
| ❌ | Inutilisable en **production** |
| ❌ | Inutilisable pour de **vrais modèles** |

> C'est un jouet éducatif, mais un jouet **indispensable** pour vraiment comprendre ce qui se passe sous le capot de PyTorch.

---

## 6. Résumé visuel

```
┌─────────────────────────────────────────────────┐
│                  MODÈLES                        │
│   Bigram → N-gram → Attention → Transformer     │
│   (ce qu'on construit)                          │
└─────────────────────────────────────────────────┘
                       ×
┌─────────────────────────────────────────────────┐
│                  OUTILS                         │
│   Micrograd → PyTorch → TensorFlow              │
│   (comment on entraîne)                         │
└─────────────────────────────────────────────────┘
```

La roadmap mélange ces deux axes dans un seul fil linéaire, ce qui crée la confusion. L'idéal est de les comprendre comme deux dimensions indépendantes du Machine Learning.
