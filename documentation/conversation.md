# Notes de conversation — Concepts NLP

---

## NLP (Natural Language Processing)

NLP = traitement du langage naturel. "Naturel" par opposition aux langages formels (Python, SQL, maths) — c'est le langage humain, ambigu, avec ses exceptions, ses métaphores, son contexte.

Le NLP couvre :
- **Comprendre** : classifier un texte, analyser le sentiment, extraire des entités
- **Générer** : traduire, résumer, répondre à des questions
- **Structurer** : analyser la syntaxe, résoudre les ambiguïtés

La règle fondamentale : une machine ne comprend que des nombres. Il faut donc toujours un système de correspondance `texte → nombre`. La vectorisation est le pont entre le texte brut et les algorithmes mathématiques.

---

## Vocabulaire vs Tokenizer

Ces deux notions sont liées mais distinctes.

**Vocabulaire** (ou dictionnaire — les deux termes sont utilisés, "vocabulaire" est plus précis en NLP) : la **liste** de tous les tokens connus. Un tableau de correspondance `fragment → numéro`.

**Tokenizer** : l'**outil/algorithme** qui découpe le texte en utilisant ce vocabulaire.

Le tokenizer *utilise* le vocabulaire pour faire son travail.

### Les tokens dans un LLM

Quand on envoie un prompt à un LLM (ex : GPT), le tokenizer découpe le texte en morceaux — par exemple `["Bon", "jour", ",", " comment", " vas", "-", "tu", " ?"]` — et le nombre de morceaux = le nombre de tokens.

Les LLMs utilisent des **sous-mots** (fragments de mots) plutôt que des mots entiers, pour deux raisons :
1. Un vocabulaire de mots entiers devrait contenir des millions d'entrées (toutes les langues, conjugaisons, etc.)
2. Un mot inconnu devient ingérable — avec des sous-mots, n'importe quel mot peut être décomposé même s'il n'a jamais été vu

### Schéma

```
Texte → Tokenizer → [token1, token2, ...] → [42, 817, ...] → Modèle
                         ↑
               utilise le vocabulaire
               pour faire la découpe
```

Les méthodes classiques (BoW, TF-IDF) sont une version simplifiée : le "tokenizer" est un simple `split()` sur les espaces, et le vocabulaire contient des mots entiers.

---

## Tokenizer et vocabulaire : une relation indissociable

En pratique, un tokenizer et son vocabulaire ne sont jamais utilisés l'un sans l'autre. Les équipes de recherche les développent ensemble, et ils sont distribués ensemble. C'est le tokenizer qui a un nom, une histoire, une réputation — pas le vocabulaire. Le vocabulaire est une sous-partie technique, interne, qui accompagne le tokenizer mais qui n'a pas d'identité propre.

Quelques tokenizers célèbres :
- **BERT** (Google) — utilise WordPiece, ~30 522 tokens
- **GPT-2 / GPT-3 / GPT-4** (OpenAI) — utilise BPE via tiktoken, ~50 257 tokens
- **Llama** (Meta) — utilise SentencePiece, ~32 000 tokens

Quand un développeur utilise un LLM aujourd'hui, il ne crée jamais son propre tokenizer. Il télécharge celui du modèle choisi et l'utilise directement. Le vocabulaire vient avec, intégré.

L'analogie qui tient : le tokenizer c'est un couteau de chef, le vocabulaire c'est sa lame. La lame est indissociable du couteau et fait partie de ce qui le définit, mais on donne un nom au couteau — pas un nom séparé à la lame.

---

## Histoire de la tokenisation

La tokenisation est un concept bien plus ancien qu'on ne l'imagine. Découper du texte en mots et compter les fréquences existe depuis les débuts de l'informatique, dans les années 1950-60. Les premiers systèmes de recherche documentaire (Information Retrieval) faisaient déjà ça. Même le concept de "stop words" — ignorer les mots vides comme "le", "la", "les" — date de cette époque.

L'algorithme BPE (Byte Pair Encoding) a été inventé en **1994** par Philip Gage, mais pour la **compression de données**, pas pour le NLP. L'idée était de remplacer les paires de bytes les plus fréquentes par un seul byte pour réduire la taille des fichiers. L'application au NLP pour créer des sous-mots est venue bien plus tard, en **2016** (papier de Sennrich et al.). L'algorithme avait donc attendu 22 ans dans un tiroir avant de trouver son usage en traitement du langage.

Les noms célèbres comme BERT ou GPT datent de **2018-2019**, avec l'explosion des Transformers.

| Époque | Ce qui existait |
|--------|----------------|
| 1950-60 | Tokenisation basique, découpe en mots, stop words |
| 1994 | BPE inventé pour la compression de données |
| 2016 | BPE appliqué au NLP pour les sous-mots |
| 2018+ | BERT, GPT — les tokenizers "célèbres" |

Ce pattern est très courant en informatique et en sciences : le machine learning n'a pas tout inventé from scratch. Il a recyclé et réassemblé des décennies de connaissances qui attendaient leur moment. BPE pour la compression, les réseaux de neurones des années 50-60 redécouverts dans les années 2000, la rétropropagation formalisée dans les années 80 et longtemps ignorée faute de puissance de calcul, le mécanisme d'attention qui avait des précurseurs bien avant le papier "Attention is All You Need" de 2017. Ce qui a tout débloqué, ce n'est pas une invention révolutionnaire d'un concept nouveau — c'est la convergence de trois choses : des algorithmes qui existaient déjà, des données massives (internet, Common Crawl...) et de la puissance de calcul (GPU, puis TPU).

---

## Ce qui fait vraiment la puissance d'un LLM

Le tokenizer et son vocabulaire sont nécessaires mais loin d'être ce qui rend un LLM puissant. C'est une brique d'entrée — un convertisseur. Ce qui fait vraiment la puissance, par ordre d'importance :

**1. L'architecture Transformer** — le vrai game changer. Le mécanisme d'attention permet au modèle de comprendre les relations entre les mots à longue distance. "Le chat que j'ai vu hier dans le jardin dort" — le modèle comprend que "dort" se rapporte à "chat" malgré la distance. Aucune architecture avant les Transformers ne faisait ça bien.

**2. La quantité de données d'entraînement** — des centaines de milliards de tokens issus du web, de livres, de Wikipedia, de code. C'est ce qui donne la "connaissance" au modèle, encodée dans ses paramètres.

**3. Le nombre de paramètres** — GPT-4 aurait environ 1 000 milliards de paramètres. C'est la capacité du modèle à stocker des patterns complexes.

**4. La puissance de calcul** — sans GPU/TPU massifs, rien de tout ça n'est possible.

**5. Le fine-tuning et RLHF** — ce qui rend les modèles agréables à utiliser, capables de suivre des instructions et de ne pas produire de contenu dangereux.

La métaphore du traitement du signal tient bien : le tokenizer est le convertisseur (signal analogique → numérique), les paramètres sont le processeur (le vrai traitement).

```
Texte humain
    → Tokenizer (conversion signal humain → numérique)
    → Paramètres / couches Transformer (traitement, intelligence)
    → Texte humain en sortie
```

---

## Architecture interne : vocabulaire, embedding, paramètres

Il faut distinguer clairement ce qui est stocké dans un LLM.

### Le vocabulaire

Concrètement en Python, le vocabulaire est stocké sous forme de **deux dictionnaires** — un dans chaque sens :

```python
# Token → ID (pour encoder le texte entrant)
token_to_id = {
    "chat": 1234,
    "chien": 5678,
    "le": 42,
    ...
}

# ID → Token (pour décoder la sortie du modèle)
id_to_token = {
    1234: "chat",
    5678: "chien",
    42: "le",
    ...
}
```

On a besoin des deux directions : à l'entrée on convertit du texte en IDs, à la sortie on reconvertit des IDs en texte. On pourrait techniquement se débrouiller avec un seul dictionnaire en bouclant dessus pour trouver l'inverse, mais ce serait très lent — parcourir 50 000 entrées à chaque token produit, des millions de fois, devient un goulot d'étranglement énorme. Deux dictionnaires permettent une réponse instantanée dans les deux sens. C'est un compromis classique en informatique : on sacrifie un peu de mémoire pour gagner beaucoup en vitesse.

En pratique chez OpenAI, tiktoken stocke ça dans un fichier `.tiktoken` encodé en base64, chargé en mémoire au démarrage.

### La matrice d'embedding

La matrice d'embedding est la première vraie "couche" du modèle. C'est une matrice 2D dont :
- La **hauteur** = taille du vocabulaire (ex : 32 000 lignes pour Llama)
- La **largeur** = dimension des embeddings (ex : 768 colonnes pour BERT)

```
                768 colonnes (dimensions)
              ←―――――――――――――――――――――――――→
    token 0   [ 0.23, -0.87,  0.41, ... ]  ↑
    token 1   [ 0.11,  0.65, -0.92, ... ]  |
    token 2   [ 0.54,  0.12,  0.33, ... ]  | 32 000 lignes
    ...                                     |
    token 32000 [ ... ]                     ↓
```

Chaque ligne est un **vecteur** — une liste de nombres à une seule dimension. Pas une matrice imbriquée, juste une liste. La hauteur du vocabulaire et la hauteur de la matrice d'embedding sont **forcément identiques** — chaque token doit avoir exactement une ligne, ni plus ni moins.

Cette matrice est **apprise pendant l'entraînement**. Au départ les valeurs sont aléatoires, et progressivement le modèle ajuste ces nombres pour que des tokens similaires aient des vecteurs proches. C'est elle qui fait le pont entre l'index (le vocabulaire) et la connaissance (les paramètres).

### Les paramètres

La matrice d'embedding fait partie des paramètres, mais ce n'est qu'une fraction. Les paramètres totaux d'un LLM incluent :

```
Paramètres totaux :
├── Matrice d'embedding
├── Couches Transformer (x N)
│   ├── Matrices d'attention (Q, K, V)
│   ├── Matrices de projection
│   └── Réseaux feed-forward
└── Matrice de sortie (décodage)
```

Pour donner une idée des proportions sur GPT-2 (~117M paramètres) :
- Matrice d'embedding : ~40 millions de paramètres
- Couches Transformer : ~77 millions de paramètres

Plus le modèle est grand, plus les couches Transformer dominent.

---

## La dimension des embeddings : plus c'est grand, mieux c'est ?

Plus de dimensions dans les vecteurs d'embedding = plus de capacité d'expression. Un vecteur de 768 dimensions peut encoder plus de nuances sur un token qu'un vecteur de 64 dimensions — comme une photo 4K contient plus d'informations qu'une photo 240p.

Mais plus de dimensions ne signifie pas forcément un modèle plus performant, pour plusieurs raisons :

- Le goulot d'étranglement n'est pas là : la performance vient surtout du nombre de couches Transformer et de leur qualité, pas de la taille des embeddings.
- Le surapprentissage : trop de dimensions sur un petit corpus, et le modèle mémorise les données d'entraînement au lieu d'apprendre des patterns généraux.
- Le coût computationnel : doubler les dimensions quadruple environ le coût de calcul. À un moment le gain ne vaut plus le coût.

Ce qui fait la vraie différence c'est l'équilibre global entre dimensions, nombre de couches, et quantité de données — pas une seule variable isolée.

| Modèle | Dimensions embedding |
|--------|---------------------|
| BERT base | 768 |
| GPT-2 | 768 |
| GPT-3 | 12 288 |

---

## La boîte noire : que représentent les 768 dimensions ?

C'est l'une des questions les plus profondes du domaine. Ces 768 nombres émergent de l'entraînement de manière automatique — personne n'a dit "la dimension 42 représentera l'animalité" ou "la dimension 107 représentera la rapidité". Le modèle a trouvé ses propres représentations seul, optimisées pour prédire des tokens.

C'est largement une boîte noire, mais pas totalement. Des chercheurs ont fait des expériences et trouvé des dimensions qui semblent correspondre à des concepts : une dimension qui s'active fortement pour les noms propres, une autre pour les verbes d'action, une autre pour les concepts géographiques. Ce domaine de recherche s'appelle l'**interpretability** (ou mechanistic interpretability) — c'est un domaine très actif en ce moment, notamment chez Anthropic.

Mais c'est rarement aussi propre qu'on l'espèrerait. La plupart du temps, un concept est réparti sur plusieurs dimensions à la fois, et une dimension participe à plusieurs concepts simultanément. Les représentations sont obliques, entrelacées. C'est ce qu'on appelle le **polysémantisme** des neurones : un neurone (ou une dimension) peut encoder plusieurs choses en même temps.

Ni totalement opaque, ni totalement lisible. Quelques patterns émergent, mais la majorité reste difficilement interprétable par un humain. C'est d'ailleurs l'un des grands défis actuels de l'IA — comprendre ce qui se passe vraiment à l'intérieur de ces modèles.

---

## Tokeniser vs Vectoriser — distinction fondamentale

**Ne jamais confondre ces deux étapes.**

**Tokeniser** = donner un ID à un token via le vocabulaire. C'est fixe, ça ne change jamais.
```
"lentement" → ID 8
```

**Vectoriser** = transformer cet ID en vecteur via la matrice d'embedding. Ce vecteur évolue pendant l'entraînement et se stabilise à la fin.
```
ID 8 → [ -0.1,  0.3,  0.4, -0.5,  0.2,  0.8 ]
```

Le token (ID) est permanent. Le vecteur est une représentation apprise — il reflète l'état de la connaissance du modèle à un instant T. Au début de l'entraînement il est aléatoire, à la fin il est stabilisé et ne bouge plus.

---

## Le processus d'entraînement — forward pass et rétropropagation

### Les trois matrices du modèle simplifié

Pour comprendre l'entraînement sans se perdre dans les Transformers, on part d'un modèle à une seule couche cachée. Trois matrices, toutes initialisées aléatoirement :

```
Matrice d'embedding  [vocab_size × embed_dim]  ex: [12 × 6]
Matrice de poids W   [embed_dim × hidden_dim]  ex: [6 × 6]
Matrice de sortie S  [hidden_dim × vocab_size] ex: [6 × 12]
```

Les contraintes dures :
- Hauteur de l'embedding = taille du vocabulaire (obligatoire)
- Colonnes de S = taille du vocabulaire (obligatoire)
- Tout le reste = choix libres d'architecture

### Le forward pass étape par étape

Exemple : `input = "lentement" (ID 8)`, `target = "nuit" (ID 9)`

```
1. Tokenisation :
   "lentement" → ID 8

2. Vectorisation :
   ID 8 → ligne 8 de la matrice d'embedding
   vecteur_entree = [ -0.1,  0.3,  0.4, -0.5,  0.2,  0.8 ]

3. Couche cachée :
   vecteur_entree [6]  ×  Matrice W [6 × 6]  =  vecteur_cache [6]

4. Matrice de sortie :
   vecteur_cache [6]  ×  Matrice S [6 × 12]  =  scores [12]
   → un score par mot du vocabulaire

5. Softmax :
   scores [12]  →  probabilités [12]  (entre 0 et 1, somme = 1)

6. Loss (cross-entropy) :
   loss = -log(probas[9])   ← probabilité donnée à "nuit"
   → plus la proba est petite, plus la loss est grande
```

### Pourquoi les positions ne se perdent pas

La position 9 dans le vecteur de scores correspond toujours à "nuit" parce que la matrice S a été construite avec 12 colonnes correspondant aux 12 mots du vocabulaire. C'est câblé avant l'entraînement — le modèle ne choisit pas cette structure, il apprend uniquement les valeurs à l'intérieur.

### La rétropropagation

On remonte en arrière via la **chain rule** (règle de la chaîne). Un seul point de départ (la loss), qui génère un gradient différent pour chaque matrice :

```
loss
  ↓  chain rule
dL/dS  → mise à jour de S
dL/dW  → mise à jour de W
dL/dE  → mise à jour de la ligne 8 dans l'embedding
```

La mise à jour de chaque paramètre :
```
parametre_nouveau = parametre_ancien - learning_rate × gradient
```

Ce qui est mis à jour à chaque tour :
- La ligne du **token en entrée** dans l'embedding
- La matrice W
- La matrice S

La ligne du **token cible** dans l'embedding n'est PAS mise à jour — elle n'a pas participé au calcul.

### La boucle complète

```python
for epoch in range(nb_epochs):
    for i in range(len(input_ids)):
        # Forward pass
        vecteur_entree = embedding[input_ids[i]]
        vecteur_cache  = vecteur_entree @ W
        scores         = vecteur_cache @ S
        probas         = softmax(scores)
        loss           = -log(probas[target_ids[i]])

        # Rétropropagation
        gradients = backprop(loss)
        S         -= learning_rate * gradients.S
        W         -= learning_rate * gradients.W
        embedding[input_ids[i]] -= learning_rate * gradients.E
```

Au premier tour tout est aléatoire — du bruit. Après des millions de tours sur des millions de textes, les matrices convergent et reflètent de vraies relations sémantiques.

---

## Transformers et attention

Un Transformer c'est fondamentalement une succession de multiplications matricielles. Ce qui le rend spécial c'est le **mécanisme d'attention**.

Dans un simple réseau, chaque token est traité indépendamment — "lentement" passe dans ses matrices sans savoir que "chat" et "chasse" sont dans la même phrase.

Le mécanisme d'attention ajoute une étape où chaque token **regarde tous les autres tokens** de la phrase et décide lesquels sont importants pour lui, avant de passer dans les matrices de poids.

```
Simple réseau  : token → W → W → W → scores
                 chaque token traité seul

Transformer    : token → attention → W → attention → W → scores
                 chaque token influencé par les autres
```

Transformer et attention sont indissociables — le Transformer est précisément l'architecture qui a fait de l'attention la pièce centrale et unique. Avant 2017, l'attention existait mais était greffée sur d'autres architectures (RNN). Le papier "Attention is All You Need" (2017) a proposé de tout jeter sauf l'attention.

---

## Word2Vec — c'est exactement l'entraînement d'une matrice d'embedding

Word2Vec n'est pas une technique mystérieuse — c'est exactement le processus d'entraînement qu'on vient de décrire, appliqué à l'apprentissage de vecteurs de mots.

Le mécanisme est identique : forward pass → loss → rétropropagation → mise à jour des matrices. La seule différence c'est la tâche d'entraînement.

### Deux variantes

**Skip-gram** : on donne un mot en entrée, on prédit les mots autour (le contexte)
```
"chat" → prédit ["petit", "chasse"]
```

**CBOW (Continuous Bag of Words)** : on donne le contexte en entrée, on prédit le mot central
```
["petit", "chasse"] → prédit "chat"
```

### Ce qu'on garde à la fin

```
Matrice d'embedding  ← on garde ça, c'est le résultat de Word2Vec
Matrice W            ← jetée après l'entraînement
Matrice de sortie S  ← jetée après l'entraînement
```

Word2Vec = "entraîne un réseau pour prédire des mots dans leur contexte, puis garde uniquement la matrice d'embedding."

### Différence avec le mécanisme d'attention

Word2Vec regarde le contexte pour **apprendre** des vecteurs fixes — "chat" aura toujours le même vecteur peu importe la phrase.

L'attention regarde le contexte pour **modifier** le vecteur d'un token en temps réel — "chat" aura un vecteur différent dans "chat dort" et dans "chat sauvage".

C'est pour ça que les Transformers comprennent la polysémie là où Word2Vec ne le peut pas.

---

## Ce projet — vectorisation vs tokenisation

Le repo s'appelle "vectorisation" — c'est le bon nom. BoW, TF-IDF, Word2Vec sont toutes des techniques de **vectorisation** : transformer du texte en vecteurs exploitables mathématiquement.

La conversation a largement débordé vers les internals des LLMs (tokenisation, entraînement, rétropropagation) mais Word2Vec (Semaine 3 de la roadmap) est directement le même mécanisme que la matrice d'embedding des LLMs — donc tout ce qui a été vu ici est pertinent pour la suite.
