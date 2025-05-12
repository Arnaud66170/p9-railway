---
title: Détection des émotions – Projet P9
emoji: 🧠
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: "3.50.2"
app_file: app.py
pinned: false
---

# 🧠 Détection des émotions (Projet P9 – OpenClassrooms)

Cette application Gradio permet d’analyser les émotions présentes dans des tweets ou messages courts.  
Elle repose sur un modèle **ELECTRA-small** fine-tuné sur le dataset **GoEmotions** (Google Research), couvrant **28 étiquettes émotionnelles multi-label**.

---

## ✨ Fonctionnalités principales

- 🔍 Prédiction multi-émotion (sigmoid) avec scores de confiance
- 📊 Visualisation dynamique : camembert, historique des 5 derniers tweets
- 🧾 Journalisation automatique (analyses + feedbacks)
- 📩 Feedback utilisateur (👍 / 👎) + commentaires
- ⚠️ Alerte mail automatique si 3 feedbacks négatifs en moins de 5 min
- 🗂️ Téléchargement des prédictions au format CSV

---

## 🔬 Modèle utilisé

- **Architecture** : ELECTRA-small discriminator
- **Type** : Transformer optimisé
- **Tâche** : `text-classification` multi-label (activation `sigmoid`)
- **Dataset** : [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
- **Nombre de labels** : 28 émotions + neutre

---

## 🛠️ Informations projet

- Projet réalisé dans le cadre de la formation **Data Scientist – OpenClassrooms**
- Ce projet P9 est une **amélioration du projet P7**, basé initialement sur TF-IDF + Logistic Regression
- Code complet et modularisé : `huggingface_api/`, `src/`, `scripts/`

---

## 🚀 Déployé avec :
- Gradio (interface utilisateur)
- Hugging Face Spaces
- MLflow (Model Registry + suivi des versions)
- Entraînement local avec GPU (GTX 1060)

---

➡️ Retrouvez le code complet dans le projet : `P9_V2`  
📂 [GitHub / Hugging Face à venir]

