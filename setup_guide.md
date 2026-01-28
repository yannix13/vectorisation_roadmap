# 🚀 Guide de Setup - Environnement Python pour la Roadmap

## 📋 Checklist rapide

- [ ] Python installé
- [ ] VS Code installé
- [ ] Extension Python VS Code
- [ ] Environnement virtuel créé
- [ ] Librairies installées
- [ ] Premier script testé

---

## 1️⃣ Installer Python

### Windows

1. Va sur https://www.python.org/downloads/
2. Télécharge Python 3.10 ou plus récent
3. **IMPORTANT** : Coche "Add Python to PATH" pendant l'installation !
4. Vérifie l'installation :

```bash
python --version
# ou
python3 --version
```

### Mac

```bash
# Avec Homebrew (recommandé)
brew install python3

# Vérifier
python3 --version
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Vérifier
python3 --version
```

---

## 2️⃣ Installer VS Code

1. Télécharge depuis https://code.visualstudio.com/
2. Installe normalement
3. Lance VS Code

---

## 3️⃣ Extensions VS Code essentielles

### Extension Python (OBLIGATOIRE)

1. Ouvre VS Code
2. Clique sur l'icône Extensions (carré avec 4 carrés) ou `Ctrl+Shift+X`
3. Cherche "Python"
4. Installe l'extension officielle de Microsoft

### Extensions bonus (recommandées)

```
- Python Indent (facilite l'indentation)
- Pylance (meilleure autocomplétion)
- Jupyter (pour notebooks si besoin plus tard)
- Error Lens (voir les erreurs inline)
- Code Runner (exécuter code rapidement)
```

---

## 4️⃣ Créer ton projet

### Structure de dossiers recommandée

```
roadmap-nlp/
│
├── semaine1-bow/
│   ├── bow_from_scratch.py
│   ├── search_engine_v01.py
│   └── tests/
│
├── semaine2-tfidf/
│   ├── tfidf_from_scratch.py
│   ├── search_engine_v02.py
│   └── tests/
│
├── semaine3-word2vec/
│   ├── word2vec_basics.py
│   ├── search_engine_v03.py
│   └── models/
│
└── data/
    └── corpus.txt
```

### Créer la structure

**Option 1 : Via terminal/cmd**

```bash
# Crée le dossier principal
mkdir roadmap-nlp
cd roadmap-nlp

# Crée les sous-dossiers
mkdir semaine1-bow semaine2-tfidf semaine3-word2vec data

# Crée un fichier test
touch semaine1-bow/bow_from_scratch.py
```

**Option 2 : Via VS Code**

1. Ouvre VS Code
2. `Fichier > Ouvrir le dossier` (ou `Ctrl+K Ctrl+O`)
3. Crée un nouveau dossier `roadmap-nlp`
4. Sélectionne ce dossier
5. Crée les sous-dossiers avec le bouton "Nouveau dossier"

---

## 5️⃣ Environnement virtuel (CRUCIAL !)

### Pourquoi ?

Un environnement virtuel isole tes librairies Python pour ce projet. Évite les conflits !

### Créer l'environnement virtuel

**Dans VS Code, ouvre un terminal :**
- `Terminal > Nouveau Terminal` (ou `` Ctrl+` ``)

**Puis tape :**

```bash
# Windows
python -m venv venv

# Mac/Linux
python3 -m venv venv
```

Cela crée un dossier `venv/` dans ton projet.

### Activer l'environnement virtuel

**Windows (CMD):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

**Tu sais que c'est activé quand tu vois `(venv)` avant ton prompt !**

```bash
(venv) C:\Users\...\roadmap-nlp>
```

### VS Code : Sélectionner l'interpréteur Python

1. `Ctrl+Shift+P` (ouvre la palette de commandes)
2. Tape : "Python: Select Interpreter"
3. Choisis celui qui contient `venv` (ex: `.\venv\Scripts\python.exe`)

---

## 6️⃣ Installer les librairies

**Avec ton environnement virtuel activé :**

### Pour Semaine 1 (BoW)

```bash
pip install numpy
```

### Pour Semaine 2 (TF-IDF)

```bash
pip install numpy scikit-learn matplotlib
```

### Pour Semaine 3 (Word2Vec)

```bash
pip install numpy scikit-learn matplotlib gensim
```

### Tout installer d'un coup (recommandé)

```bash
pip install numpy scikit-learn matplotlib gensim nltk
```

### Créer un fichier requirements.txt

```bash
pip freeze > requirements.txt
```

Plus tard, tu pourras réinstaller tout avec :

```bash
pip install -r requirements.txt
```

---

## 7️⃣ Tester l'installation

### Créer un fichier test.py

```python
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
```

### Exécuter

**Option 1 : Dans le terminal**
```bash
python test.py
```

**Option 2 : Dans VS Code**
- Clique droit sur le fichier > "Run Python File in Terminal"
- Ou appuie sur le bouton ▶️ en haut à droite

**Résultat attendu :**
```
✅ NumPy version: 1.24.x
✅ Sklearn fonctionne!
✅ Matrice TF-IDF shape: (2, 4)

🎉 Tout est prêt ! Tu peux commencer la roadmap.
```

---

## 8️⃣ Configuration VS Code (optionnel mais utile)

### Créer .vscode/settings.json

Crée un dossier `.vscode` dans ton projet, puis un fichier `settings.json` :

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "autopep8",
    "editor.formatOnSave": true,
    "python.terminal.activateEnvironment": true,
    "files.autoSave": "afterDelay",
    "files.autoSaveDelay": 1000
}
```

### Créer .gitignore (si tu utilises Git)

```
# Environnement virtuel
venv/
env/

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.pytest_cache/

# VS Code
.vscode/
.DS_Store

# Modèles (gros fichiers)
*.model
*.bin
```

---

## 9️⃣ Snippets utiles pour VS Code

### Créer des snippets personnalisés

1. `Fichier > Préférences > Configurer les extraits de code utilisateur`
2. Choisis "Python"
3. Ajoute :

```json
{
    "Python Script Header": {
        "prefix": "pyheader",
        "body": [
            "# ${1:filename}.py",
            "# ${2:Description}",
            "",
            "import numpy as np",
            "",
            "def main():",
            "    ${3:pass}",
            "",
            "",
            "if __name__ == '__main__':",
            "    main()"
        ],
        "description": "Header for Python scripts"
    }
}
```

Maintenant, tape `pyheader` et appuie sur Tab pour avoir un template !

---

## 🔟 Premier script : bow_from_scratch.py

Crée ton premier fichier et colle ce code de démarrage :

```python
# semaine1-bow/bow_from_scratch.py
# Implémentation de Bag of Words from scratch

def create_vocabulary(documents):
    """
    Crée le vocabulaire à partir d'une liste de documents
    
    Args:
        documents: liste de strings ["doc1", "doc2", ...]
    
    Returns:
        liste de mots uniques, triée alphabétiquement
    """
    # TODO : À compléter !
    all_words = []
    for doc in documents:
        words = doc.lower().split()
        all_words.extend(words)
    
    vocabulary = sorted(set(all_words))
    return vocabulary


def document_to_vector(document, vocabulary):
    """
    Transforme un document en vecteur BoW
    
    Args:
        document: string
        vocabulary: liste de mots
    
    Returns:
        liste de nombres (vecteur)
    """
    # TODO : À compléter !
    pass


def main():
    """
    Fonction principale pour tester
    """
    # Test documents
    documents = [
        "le chat dort sur le tapis",
        "le chien aboie dans le jardin",
        "le chat mignon joue"
    ]
    
    # Créer vocabulaire
    vocab = create_vocabulary(documents)
    print("Vocabulaire:", vocab)
    print(f"Taille: {len(vocab)} mots")


if __name__ == "__main__":
    main()
```

### Exécuter

```bash
python semaine1-bow/bow_from_scratch.py
```

---

## 🐛 Problèmes courants

### "python n'est pas reconnu..."

**Solution :** Python n'est pas dans le PATH.
- Windows : Réinstalle Python et coche "Add to PATH"
- Ou utilise `python3` au lieu de `python`

### "pip n'est pas reconnu..."

```bash
python -m pip install [package]
```

### L'environnement virtuel ne s'active pas (PowerShell)

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### VS Code n'utilise pas le bon Python

1. `Ctrl+Shift+P`
2. "Python: Select Interpreter"
3. Choisis celui dans `venv/`

### Import ne fonctionne pas

Vérifie que :
1. L'environnement virtuel est activé (`(venv)` visible)
2. Les packages sont installés : `pip list`
3. VS Code utilise le bon interpréteur

---

## ✅ Checklist finale

Avant de commencer la roadmap, vérifie que tu as :

- [ ] Python fonctionne : `python --version`
- [ ] VS Code installé avec extension Python
- [ ] Environnement virtuel créé et activé
- [ ] Librairies installées : `pip list` montre numpy, sklearn, etc.
- [ ] Structure de dossiers créée
- [ ] `test.py` s'exécute sans erreur
- [ ] Premier script `bow_from_scratch.py` créé

---

## 🚀 Tu es prêt !

Tu peux maintenant :
1. Ouvrir la roadmap (avec Obsidian/Typora/StackEdit)
2. Ouvrir VS Code avec ton projet
3. Commencer **SEMAINE 1 - JOUR 1** !

### Commandes essentielles à retenir

```bash
# Activer environnement
# Windows CMD:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# Installer une librairie
pip install [nom-package]

# Exécuter un script
python mon_script.py

# Désactiver environnement
deactivate
```

---

## 📚 Ressources supplémentaires

- **Documentation Python :** https://docs.python.org/fr/3/
- **VS Code Python Tutorial :** https://code.visualstudio.com/docs/python/python-tutorial
- **Real Python (tutoriels) :** https://realpython.com/

---

## 💡 Conseils

1. **Ouvre toujours VS Code via le terminal** après avoir activé l'environnement
2. **Sauvegarde régulièrement** (VS Code le fait automatiquement si configuré)
3. **Utilise Git** pour versioner ton code (optionnel mais recommandé)
4. **Teste après chaque fonction** écrite
5. **N'hésite pas à utiliser `print()` partout** pour déboguer !

---

Bonne chance pour la roadmap ! 🎉

Si tu rencontres un problème, note exactement :
- La commande que tu as tapée
- Le message d'erreur complet
- Ce que tu as déjà essayé

Et on pourra déboguer ensemble ! 😊
