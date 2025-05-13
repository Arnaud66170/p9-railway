# huggingface_api/app.py

# app.py (version Railway)

import gradio as gr
import sys
import os
sys.path.append(os.path.abspath("src"))

import pandas as pd
import plotly.express as px
from collections import deque
from datetime import datetime, timedelta
import csv
import random
import threading
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from utils.logger import log_user_event
from utils.alert_email import send_alert_email

# === Chargement du modèle LOCAL (ELECTRA fine-tuné) ===
MODEL_DIR = "models/electra_model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None,  # multi-label
    function_to_apply="sigmoid"
)

# === Globals ===
HISTORY_LIMIT = 5
history = deque(maxlen=HISTORY_LIMIT)
ALERT_WINDOW_MINUTES = 5
ALERT_COOLDOWN_MINUTES = 10
FEEDBACK_ALERT_THRESHOLD = 3
alert_history = []
FEEDBACK_CSV = os.path.abspath("feedback_log_emotions.csv")

# === Données ===
tweet_examples = [
    "I feel amazing today!", "I'm so angry right now.", "Everything is fine, just a little tired.",
    "Why is nobody answering me?", "This is the best day of my life!", "Sure, because waiting 3 hours for coffee is totally normal 🚒",
    "I miss you... but maybe it's for the best.", "Finally got that promotion 🥳", "Just great. Another Monday. Yay.",
    "Thank you for ruining my day 😊", "She smiled, but her eyes were empty.", "I can’t take this anymore.",
    "That’s it. I’m done. Don’t call me.", "Well... that escalated quickly.", "I'm so proud of you ❤️",
    "Guess who forgot their keys again 🙄", "I just want to disappear for a while.", "Wow. Thanks a lot. Really helpful. 👏",
    "I can't believe you remembered 🥺", "Oh joy, another pointless meeting...", "Let's pretend everything’s okay 🌪️",
    "Honestly? I don’t even care anymore.", "You’ve made my day ☀️", "Love this song. Hits hard today.",
    "So peaceful here. I could stay forever.", "You always know just what to say 😌", "I guess it’s whatever now 🤷",
    "What’s the point of trying?", "I laughed so hard I cried 😂", "This meal is a 10/10 👌",
    "Please don’t leave me.", "I knew this would happen.", "Don’t talk to me like that again.",
    "You forgot again. Of course you did.", "Oh wow, a surprise. Totally didn’t expect that 🤭",
    "That was unexpected... and kinda sweet.", "Creeped out. That guy followed me home.", "Ugh. Can't stand this anymore.",
    "So tired of pretending I'm fine.", "Feeling super grateful today 🙏", "Can’t stop smiling 😁",
    "I'm proud of how far I’ve come.", "Back at it again. Let’s gooo 💪", "I’m trying, okay? I really am.",
    "This is the dumbest thing ever.", "You always ruin everything 😡", "Missing the good old days.",
    "Best birthday ever 🎂🎈", "I'm not crying. You are 😢", "Guess I should’ve seen it coming.",
    "No one listens. No one cares."
]

LABEL_MAP = {
    f"LABEL_{i}": label for i, label in enumerate([
        "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
        "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
        "remorse", "sadness", "surprise", "neutral"
    ])
}

# === Fonctions ===
def predict_emotions(text):
    outputs = pipeline(text)
    print("\U0001f50d pipeline raw output:", outputs)

    if isinstance(outputs, list):
        emotion_scores = outputs[0] if isinstance(outputs[0], list) else outputs
    else:
        return "❌ Sortie invalide du pipeline.", None, None

    sorted_emotions = sorted(emotion_scores, key=lambda x: x['score'], reverse=True)
    dominant = sorted_emotions[0]
    label_readable = LABEL_MAP.get(dominant['label'], dominant['label'])
    history.appendleft({"text": text, "emotion": label_readable, "score": round(dominant['score']*100, 2)})

    return f"<h2 style='text-align:center;'>🧠 Emotion dominante : {label_readable.capitalize()} ({round(dominant['score']*100)}%)</h2>", update_pie_chart(), update_history()

def update_pie_chart():
    emotions = [h['emotion'] for h in history]
    df = pd.DataFrame(emotions, columns=['emotion'])
    if df.empty:
        return px.pie(names=[], values=[])
    counts = df['emotion'].value_counts().reset_index()
    counts.columns = ['Emotion', 'Count']
    return px.pie(counts, names='Emotion', values='Count', title='Distribution des émotions récentes')

def update_history():
    df = pd.DataFrame(list(history))
    return df.rename(columns={"text": "Tweet", "emotion": "Emotion", "score": "Confiance (%)"}) if not df.empty else pd.DataFrame(columns=["Tweet", "Emotion", "Confiance (%)"])

def save_feedback(tweet, feedback, comment):
    if not history:
        return "❌ Aucun historique disponible.", ""

    last = history[0]
    emotion = last["emotion"]
    confidence = last["score"]
    timestamp = datetime.now()
    row = {
        "tweet": tweet,
        "predicted_emotion": emotion,
        "proba": confidence,
        "user_feedback": feedback,
        "comment": comment,
        "timestamp": timestamp.isoformat()
    }
    try:
        file_exists = os.path.exists(FEEDBACK_CSV)
        with open(FEEDBACK_CSV, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"Erreur CSV : {e}")

    log_user_event("feedback", tweet_text=tweet, predicted_label=emotion, proba=confidence, feedback=feedback, comment=comment)

    if feedback == "👎 No":
        alert_history.append(timestamp)
        now = datetime.now()
        recent_alerts = [t for t in alert_history if now - t < timedelta(minutes=ALERT_WINDOW_MINUTES)]
        alert_history[:] = recent_alerts
        if len(recent_alerts) >= FEEDBACK_ALERT_THRESHOLD:
            if not hasattr(save_feedback, "last_alert") or now - save_feedback.last_alert > timedelta(minutes=ALERT_COOLDOWN_MINUTES):
                threading.Thread(target=send_alert_email, args=(len(recent_alerts),), daemon=True).start()
                save_feedback.last_alert = now

    return "✅ Feedback enregistré.", update_feedback_stats()

def update_feedback_stats():
    if not os.path.exists(FEEDBACK_CSV):
        return "Aucun feedback encore."
    try:
        df = pd.read_csv(FEEDBACK_CSV)
        count_yes = (df['user_feedback'] == '👍 Yes').sum()
        count_no = (df['user_feedback'] == '👎 No').sum()
        return f"👍 Yes: {count_yes} | 👎 No: {count_no} | Total: {len(df)}"
    except:
        return "Erreur lecture stats."

def download_emotion_stats():
    csv_path = "outputs/emotion_stats_2000.csv"
    return csv_path if os.path.exists(csv_path) else None

# === Interface Gradio ===
with gr.Blocks() as demo:
    gr.Markdown("""
    # 😶‍🌫️ Analyse des émotions (tweets)
    """)
    with gr.Row():
        tweet_input = gr.Textbox(label="💬 Tweet", lines=3)
        example_btn = gr.Button("🎲 Exemple aléatoire")
    with gr.Row():
        analyze_btn = gr.Button("🔍 Analyser")
        sentiment_output = gr.HTML()
    pie_plot = gr.Plot()
    history_display = gr.Dataframe()
    with gr.Accordion("📩 Feedback utilisateur", open=False):
        feedback = gr.Radio(["👍 Yes", "👎 No"], label="Prédiction correcte ?")
        comment = gr.Textbox(label="Commentaire optionnel")
        feedback_btn = gr.Button("✅ Envoyer Feedback")
        feedback_status = gr.Textbox(label="Statut", interactive=False)
        feedback_stats = gr.Textbox(label="Stats feedback", interactive=False)
    with gr.Accordion("📁 Export / Logs", open=False):
        gen_button = gr.Button("📤 Télécharger CSV émotions")
        download_btn = gr.File(label="Fichier CSV", interactive=True)

    # === Actions ===
    analyze_btn.click(fn=predict_emotions, inputs=tweet_input, outputs=[sentiment_output, pie_plot, history_display])
    example_btn.click(fn=lambda: random.choice(tweet_examples), outputs=tweet_input)
    feedback_btn.click(fn=save_feedback, inputs=[tweet_input, feedback, comment], outputs=[feedback_status, feedback_stats])
    gen_button.click(fn=download_emotion_stats, outputs=download_btn)

if __name__ == "__main__":
    demo.launch()


# Appel script:
# python huggingface_api/app.py



# cd huggingface_api
# python huggingface_api/app.py


# import gradio as gr
# import os
# import mlflow

# # Chargement depuis le Model Registry (version en Production)
# model_uri = "models:/emotions_classifier/Production"
# pipeline = mlflow.transformers.load_model(model_uri)

# # Exemple : prédiction dans l'interface
# def predict_emotions(text):
#     outputs = pipeline(text)
#     return outputs

# def download_emotion_stats():
#     csv_path = "outputs/emotion_stats_2000.csv"
#     if os.path.exists(csv_path):
#         return csv_path
#     else:
#         return None

# with gr.Blocks() as demo:
#     with gr.Tab("📊 Analyse émotionnelle"):
#         gr.Markdown("### 📁 Télécharger le rapport des émotions prédites sur 2000 tweets")
#         gen_button = gr.Button("Générer & Télécharger le CSV")
#         download_btn = gr.File(label="Fichier CSV", interactive=True)

#         gen_button.click(
#             fn = download_emotion_stats,
#             inputs=[],
#             outputs=download_btn
#         )

# # Lancement de l'interface
# if __name__ == "__main__":
#     demo.launch()
