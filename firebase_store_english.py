from __future__ import annotations
import os, json, time, uuid, datetime as dt
from typing import Any, Dict, Optional

import firebase_admin
from firebase_admin import credentials, firestore

def _get_db():
    if not firebase_admin._apps:
        path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_ENGLISH")
        cred = credentials.Certificate(path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

def _ensure_session(ss: dict) -> str:
    if "session_id" not in ss:
        ss["session_id"] = str(uuid.uuid4())
        ss["session_started_at"] = int(time.time())
    return ss["session_id"]

def save_participant_snapshot(ss: dict, *, collection: str) -> str:
    """Upsert summary for this participant and append an event."""
    db = _get_db()
    session_id = _ensure_session(ss)
    now = dt.datetime.utcnow()

    profile = ss.get("profile", {}) or {}
    opinions = profile.get("opinions", {}) or {}

    doc = {
        "session_id": session_id,
        "stage": ss.get("stage", ""),
        "updated_at": now,

        "nickname": profile.get("nickname", "") or f"anon-{session_id[:8]}",
        "age_range": profile.get("age_range", ""),
        "gender": profile.get("gender", ""),
        "education": profile.get("education", ""),
        "opinions": opinions,

        # chat summaries
        **{f"chat1_{k}_blm": v for k, v in _bundle_chat_summary(ss, "chat1_messages_blm", "blm").items()},
        **{f"chat1_{k}_guns": v for k, v in _bundle_chat_summary(ss, "chat1_messages_guns", "guns").items()},
        **{f"chat1_{k}_samesex": v for k, v in _bundle_chat_summary(ss, "chat1_messages_samesex", "samesex").items()},
        **{f"chat2_{k}_blm": v for k, v in _bundle_chat_summary(ss, "chat2_messages_blm", "blm").items()},
        **{f"chat2_{k}_guns": v for k, v in _bundle_chat_summary(ss, "chat2_messages_guns", "guns").items()},
        **{f"chat2_{k}_samesex": v for k, v in _bundle_chat_summary(ss, "chat2_messages_samesex", "samesex").items()},
        
        # surveys (whatever you stored)
        "survey1_post": ss.get("survey_1", {}),
        "survey2_post": ss.get("survey_2", {}),
        "survey_final": ss.get("survey_finish", {}),

        "system_prompt_original_blm": ss.get("system_prompt_chat1_blm") if ss.get("system_prompt_chat1_blm") is not None else None,
        "system_prompt_original_guns": ss.get("system_prompt_chat1_guns") if ss.get("system_prompt_chat1_guns") is not None else None,
        "system_prompt_original_samesex": ss.get("system_prompt_chat1_samesex") if ss.get("system_prompt_chat1_samesex") is not None else None,
        "system_prompt_positive_blm": ss.get("system_prompt_chat2_blm") if ss.get("system_prompt_chat2_blm") is not None else None,
        "system_prompt_positive_guns": ss.get("system_prompt_chat2_guns") if ss.get("system_prompt_chat2_guns") is not None else None,
        "system_prompt_positive_samesex": ss.get("system_prompt_chat2_samesex") if ss.get("system_prompt_chat2_samesex") is not None else None,
    }

    doc_ref = db.collection(collection).document(session_id)
    doc_ref.set(doc, merge=True)

    # append an immutable event
    doc_ref.collection("events").add({
        "stage": doc["stage"],
        "payload": doc,
        "ts": firestore.SERVER_TIMESTAMP,
    })

    return session_id

def _bundle_chat_summary(ss, prefix: str, topic: str) -> Dict[str, Any]:
    """Collect chat metrics + arrays for a given chat prefix (chat1/chat2)."""
    utox  = ss.get(f"{prefix}_user_toxicity_{topic}", []) or []
    atox  = ss.get(f"{prefix}_assistant_toxicity_{topic}", []) or []

    return {
        "user_toxicity_mean": float(sum(utox)/len(utox)) if utox else None,
        "assistant_toxicity_mean": float(sum(atox)/len(atox)) if atox else None,
        "user_toxicity_max": float(max(utox)) if utox else None,
        "assistant_toxicity_max": float(max(atox)) if atox else None,
        "user_toxicity": utox,
        "assistant_toxicity": atox,
    }

def save_chat_transcript(
    ss: dict,
    *,
    session_id: Optional[str] = None,
    collection: str,
    chat_slot: str,                  # "chat1" | "chat2"
    topic: str,                      # "blm" | "guns" | "samesex" | ...
    split_by_slot: bool = True,      # True => separate subcollections per chat slot
) -> None:
    """
    Writes one chat transcript to Firestore.

    split_by_slot = False:
        participants/{session}/chats/{chat_slot}_{topic}
    split_by_slot = True (recommended):
        participants/{session}/chats/{chat_slot}/topics/{topic}
    """
    db = _get_db()
    if not session_id:
        session_id = _ensure_session(ss)

    # pull from your Streamlit session_state
    msgs  = ss.get(f"{chat_slot}_messages_{topic}", []) or []
    utox  = ss.get(f"{chat_slot}_user_toxicity_{topic}", []) or []
    atox  = ss.get(f"{chat_slot}_assistant_toxicity_{topic}", []) or []

    payload = {
        "chat_slot": chat_slot,
        "topic": topic,
        "message_count": len(msgs),
        "messages": msgs,
        "user_toxicity": [float(x) for x in utox],
        "assistant_toxicity": [float(x) for x in atox],
        "saved_at": firestore.SERVER_TIMESTAMP,
    }

    if split_by_slot:
        # participants/{session}/chats/{chat_slot}/topics/{topic}
        ref = (db.collection(collection)
                 .document(session_id)
                 .collection("chats")
                 .document(chat_slot)           # chat header for this slot
                 .collection("topics")
                 .document(topic))
    else:
        # participants/{session}/chats/{chat_slot}_{topic}
        ref = (db.collection(collection)
                 .document(session_id)
                 .collection("chats")
                 .document(f"{chat_slot}_{topic}"))

    ref.set(payload, merge=True)

def save_into_firebase(ss: dict, *, collection: str = "english_participants"):
    save_participant_snapshot(ss=ss, collection=collection)
    
    topics = ["blm", "guns", "samesex"]
    slots = ["chat1", "chat2"]

    for s in slots:
        for t in topics:
            save_chat_transcript(ss, session_id=ss["session_id"], collection=collection, chat_slot=s, topic=t, split_by_slot=True)