# Guide d'utilisation des fichiers CSV avec l'interface GUI

## 🚀 Démarrage rapide

1. **Lancer l'interface GUI** :
   ```bash
   python main_gui.py
   ```

2. **Charger un fichier CSV** :
   - Allez dans l'onglet "Training"
   - Cliquez sur le bouton "Load CSV" dans la section "Data Loading"
   - Naviguez vers le dossier `datasets/`
   - Sélectionnez un des fichiers CSV créés
   - Le nombre d'échantillons chargés s'affichera sous les boutons

## 📊 Fichiers CSV disponibles

### Pour débuter (problèmes simples) :
- **linear_classification.csv** : Classification binaire linéaire (2 features)
- **and_gate.csv** : Porte logique AND
- **or_gate.csv** : Porte logique OR

### Problèmes intermédiaires :
- **moons_classification.csv** : Forme de lunes (non-linéaire)
- **circles_classification.csv** : Cercles concentriques
- **blobs_classification.csv** : Clusters séparés (4 classes)

### Problèmes avancés :
- **spiral_classification.csv** : Spirale (très non-linéaire)
- **complex_classification.csv** : 5 features, 2 classes
- **multiclass_classification.csv** : 3 features, 3 classes

### Régression :
- **regression.csv** : Régression linéaire
- **sine_regression.csv** : Fonction sinus

## 🔧 Configuration recommandée par type de problème

### 1. Problèmes linéaires (linear_classification, and_gate, or_gate)
```
Architecture :
- Input size : 2
- Layer 1 : 1 neuron, sigmoid

Entraînement :
- Learning rate : 0.1
- Epochs : 100
```

### 2. Problèmes non-linéaires simples (moons, circles)
```
Architecture :
- Input size : 2
- Layer 1 : 4-8 neurons, relu
- Layer 2 : 1 neuron, sigmoid

Entraînement :
- Learning rate : 0.01
- Epochs : 200-500
```

### 3. Problèmes complexes (spiral, complex_classification)
```
Architecture :
- Input size : 2 ou 5 (selon le dataset)
- Layer 1 : 16-32 neurons, relu
- Layer 2 : 8-16 neurons, relu
- Layer 3 : 1 neuron, sigmoid

Entraînement :
- Learning rate : 0.001
- Epochs : 500-1000
```

### 4. Multi-classes (multiclass, blobs)
```
Architecture :
- Input size : 2 ou 3
- Layer 1 : 8-16 neurons, relu
- Layer 2 : nombre de classes, sigmoid

Entraînement :
- Learning rate : 0.01
- Epochs : 200-500
```

### 5. Régression
```
Architecture :
- Input size : 1 ou 2
- Layer 1 : 8-16 neurons, relu
- Layer 2 : 1 neuron, linear

Entraînement :
- Learning rate : 0.001
- Epochs : 500-1000
```

## 📝 Étapes complètes pour tester

1. **Charger les données** :
   - Onglet "Training" → "Load CSV"
   - Le système divise automatiquement en 80% train / 20% test

2. **Construire le réseau** :
   - Onglet "Architecture"
   - Définir "Input Size" selon le dataset
   - Ajouter les couches une par une
   - Cliquer "Build Network"

3. **Entraîner** :
   - Onglet "Training"
   - Configurer les hyperparamètres
   - Cliquer "Start Training"
   - Observer la progression dans le log

4. **Tester** :
   - Onglet "Testing"
   - Cliquer "Test on Dataset" pour évaluer sur les données de test
   - Ou entrer des valeurs manuelles dans "Test Input"

5. **Visualiser** :
   - Onglet "Visualization"
   - "Plot Architecture" : voir la structure du réseau
   - "Plot Training History" : voir l'évolution de l'entraînement
   - "Plot Decision Boundary" : voir les frontières de décision (2D seulement)

## 💡 Conseils

- Commencez avec des problèmes simples (linear, and_gate)
- Augmentez progressivement la complexité
- Si l'accuracy stagne, essayez :
  - D'ajouter plus de neurones/couches
  - De changer la fonction d'activation
  - D'ajuster le learning rate
  - D'augmenter les epochs

## 🎯 Objectifs d'accuracy typiques

- Problèmes linéaires : > 95%
- Moons/Circles : > 90%
- Spiral : > 85%
- Multi-classes : > 80%
- Régression : Loss < 0.1

Bon apprentissage ! 🚀