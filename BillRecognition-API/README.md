# Bill Recognition API

API backend pour reconnaître les billets de banque avec un modèle TensorFlow.

## Installation

1. Installez Python 3.9+
2. Installez les dépendances :

```bash
pip install -r requirements.txt
```

## Configuration

1. Copiez votre modèle `my_banknote_model.h5` dans le dossier racine ou mettez à jour le chemin dans `app.py`

2. Personnalisez les labels des billets dans `app.py` :

```python
BILL_LABELS = {
    0: "1 USD",
    1: "5 USD",
    # ... ajouter les vôtres
}
```

## Lancer l'API

```bash
python app.py
```

L'API démarre sur `http://localhost:5000`

## Endpoints

### 1. GET /health

Vérife que l'API fonctionne

**Réponse (200):**

```json
{
  "status": "ok",
  "model_loaded": true,
  "message": "API Bill Recognition prête"
}
```

### 2. POST /predict

Prédit le billet à partir d'une image

**Request:**

- Content-Type: `multipart/form-data`
- Fichier: `file` (JPG, PNG, GIF)

**Response Success (200):**

```json
{
  "result": "100 USD",
  "confidence": 0.95,
  "class": 5
}
```

**Response Error (400/500):**

```json
{
  "error": "Format image non autorisé"
}
```

### 3. GET /model-info

Retourne les infos du modèle

**Response (200):**

```json
{
    "model_loaded": true,
    "input_shape": "(None, 224, 224, 3)",
    "output_shape": "(None, 12)",
    "classes": 12,
    "labels": { "0": "1 USD", ... }
}
```

## Tester avec cURL

```bash
# Vérifier l'API
curl http://localhost:5000/health

# Prédire
curl -X POST -F "file=@photo.jpg" http://localhost:5000/predict
```

## Configurer l'app React Native

Dans [App.js](../bill-recognition-v2/app/index.js), remplacez l'URL :

```javascript
const response = await fetch('http://YOUR_PC_IP:5000/predict', {
```

Exemple:

- Local: `http://localhost:5000`
- Réseau: `http://192.168.1.100:5000`

## Déploiement

Pour déployer en production, utilisez Gunicorn :

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

Created for Bill Recognition with AI 🤖💵
