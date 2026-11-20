import os
from datetime import datetime, timezone
import certifi
import numpy as np
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# ---------------------------------------------------------
# MongoDB Atlas Cloud Connection Configuration
# ---------------------------------------------------------
MONGO_URI = "mongodb+srv://nithishkumarv1704_db_user:nithish123@cluster0.wg7h2xx.mongodb.net/insect_bite_db?authSource=admin&retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "insect_bite_db"
COLLECTION_NAME = "diagnosis_logs"


class BiteDBManager:
    """
    Manages MongoDB persistence for insect bite diagnostic records,
    longitudinal progression tracking, and patient query history.
    """

    def __init__(self, uri: str = MONGO_URI, db_name: str = DB_NAME):
        try:
            self.client = MongoClient(
                uri,
                tlsCAFile=certifi.where(),
                tlsAllowInvalidCertificates=True,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000,
            )
            # Verify live connection
            self.client.admin.command('ping')
            self.db = self.client[db_name]
            self.collection = self.db[COLLECTION_NAME]
            self.is_connected = True
        except (ConnectionFailure, OperationFailure, Exception) as e:
            print(f"[MongoDB Error] Connection failed: {e}")
            self.client = None
            self.db = None
            self.collection = None
            self.is_connected = False

    def log_prediction(
        self,
        user_id: str,
        predicted_class: str,
        confidence: float,
        location: str,
        symptoms: list,
        timing: str,
        raw_probs: dict,
    ) -> str:
        """Stores every inference result with a UTC timestamp."""
        if not self.is_connected or self.collection is None:
            print("[MongoDB Warning] Not connected. Record skipped.")
            return None

        record = {
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc),
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "location": location,
            "symptoms": symptoms if isinstance(symptoms, list) else list(symptoms),
            "timing": timing,
            "raw_probabilities": {str(k): float(v) for k, v in raw_probs.items()},
        }

        try:
            result = self.collection.insert_one(record)
            return str(result.inserted_id)
        except Exception as e:
            print(f"[MongoDB Error] Write operation failed: {e}")
            return None

    def get_latest_bite(self, user_id: str, location: str) -> dict:
        """Finds the most recent logged diagnosis for this user and body location."""
        if not self.is_connected or self.collection is None:
            return None

        try:
            return self.collection.find_one(
                {"user_id": user_id, "location": location},
                sort=[("timestamp", -1)],
            )
        except Exception as e:
            print(f"[MongoDB Error] Query failed: {e}")
            return None

    def get_user_history(self, user_id: str, limit: int = 10) -> list:
        """Retrieves past diagnostic records for a specific patient."""
        if not self.is_connected or self.collection is None:
            return []

        try:
            cursor = (
                self.collection.find({"user_id": user_id})
                .sort("timestamp", -1)
                .limit(limit)
            )
            return list(cursor)
        except Exception as e:
            print(f"[MongoDB Error] History fetch failed: {e}")
            return []

    def analyze_progression(
        self,
        user_id: str,
        current_class: str,
        current_location: str,
        current_symptoms: list,
    ) -> dict:
        """Analyzes whether a bite at the same location is worsening over time."""
        if not self.is_connected or self.collection is None:
            return None

        previous_record = self.get_latest_bite(user_id, current_location)
        if not previous_record:
            return None

        prev_time = previous_record.get("timestamp")
        if prev_time is None:
            return None

        if prev_time.tzinfo is None:
            prev_time = prev_time.replace(tzinfo=timezone.utc)

        hours_elapsed = (
            datetime.now(timezone.utc) - prev_time
        ).total_seconds() / 3600.0

        risk_alerts = []
        is_same_class = previous_record.get("predicted_class") == current_class

        worsening_symptoms = [
            "Expanding bullseye / target ring rash",
            "Two distinct fang marks / Central necrosis",
            "Pustule or white fluid-filled blister",
        ]

        prev_symptoms = previous_record.get("symptoms", [])
        new_severe_symptoms = [
            s
            for s in current_symptoms
            if s in worsening_symptoms and s not in prev_symptoms
        ]

        if new_severe_symptoms:
            risk_alerts.append(
                f"Worsening clinical signs detected: {', '.join(new_severe_symptoms)} appeared since last scan."
            )

        if (
            hours_elapsed > 24
            and is_same_class
            and current_class in ["ticks", "spiders"]
        ):
            risk_alerts.append(
                f"Persistence alert: Lesion active for >{hours_elapsed:.1f} hrs without resolution. Monitor for Lyme/Necrosis."
            )

        return {
            "is_followup": True,
            "previous_class": previous_record.get("predicted_class"),
            "previous_confidence": previous_record.get("confidence"),
            "hours_elapsed": hours_elapsed,
            "is_same_class": is_same_class,
            "risk_alerts": risk_alerts,
        }