# P9_V2 – Analyse d’Émotions avec ELECTRA (Gradio App)

Cette application permet de détecter les émotions principales dans un texte court (ex : tweet) à l’aide d’un modèle **ELECTRA fine-tuné sur GoEmotions**.

Interface déployée avec **Gradio**, prête à l’usage sur Render.

---

## Fonctionnalités

- ✅ Prédiction multi-label d’émotions (28 classes + neutral)
- ✅ Affichage de l’émotion dominante + score de confiance
- ✅ Historique des 5 dernières analyses
- ✅ Graphique circulaire (camembert) des émotions récentes
- ✅ Feedback utilisateur (👍/👎 + commentaire)
- ✅ Journalisation CSV (log_analysis, log_feedbacks)
- ✅ Système d’alerte e-mail si 3 feedbacks négatifs en <5 minutes
- ✅ Export des logs via bouton de téléchargement

---

## Structure du projet (résumé)

├── huggingface_space/
│ ├── app.py # Interface Gradio
│ ├── requirements.txt # Dépendances
│ ├── models/electra_model/ # Modèle ELECTRA fine-tuné
├── src/
│ └── utils/
│ ├── logger.py # Gestion des logs CSV
│ └── alert_email.py # Microservice d’alerte (Railway ou SendGrid)
├── render.yaml # Déploiement Render (automatisé)

