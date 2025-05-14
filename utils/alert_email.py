# src/utils/alert_email.py

import os
import requests

RAILWAY_ALERT_API = os.environ.get("RAILWAY_ALERT_API", "https://web-production-662f7.up.railway.app/send-alert")

def send_alert_email(nb_bad_feedbacks):
    """
    Envoie une alerte via l'API Railway quand trop de feedbacks émotionnels négatifs sont reçus.
    L'URL est configurable via la variable d'environnement RAILWAY_ALERT_API.
    """
    if not RAILWAY_ALERT_API:
        print("⚠️ Aucune URL d’alerte définie. Alerte non envoyée.")
        return

    try:
        response = requests.post(
            RAILWAY_ALERT_API,
            json={"nb_feedbacks": nb_bad_feedbacks},
            timeout=5
        )
        print(f"✅ Appel API Railway : {response.status_code} | {response.json()}")
    except Exception as e:
        print(f"❌ Échec appel API d’alerte : {e}")


# import requests

# def send_alert_email(nb_bad_feedbacks):
#     """
#     Envoie une alerte via l'API Railway quand trop de feedbacks émotionnels négatifs sont reçus.
#     """
#     try:
#         response = requests.post(
#             "https://web-production-662f7.up.railway.app/send-alert",
#             json={"nb_feedbacks": nb_bad_feedbacks}
#         )
#         print(f"✅ Appel API Railway : {response.status_code} | {response.json()}")
#     except Exception as e:
#         print(f"❌ Échec appel API d’alerte : {e}")