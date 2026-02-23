# first copy: & "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8501
# the shared link is in the +----+ box that been printed
# then copy: streamlit run chat_english.py
import os
import random
import time
from dotenv import load_dotenv
import streamlit as st
from openai import AzureOpenAI, OpenAI
from typing import List, Optional
from toxicity import measuring_toxicity
from opposite_english_nli_gpt import run_opposite_pipeline_and_render, load_english
from firebase_store_english import save_into_firebase

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is missing. Add it to a .env file or your environment.")
    st.stop()
endpoint = os.getenv("ENDPOINT_URL", "https://ai-asolomon28262ai165132345402.openai.azure.com/")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

# Initialize Azure OpenAI client with key-based authentication
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",
)

@st.cache_resource(show_spinner=False)
def load_english_cached():
    #returns meta, embs, index, encoder
    return load_english(
        "./english",
        repo_id="Liran73/english-opposite-artifacts",
        repo_type="dataset",
        hf_token_env="HF_TOKEN",
    )

st.set_page_config(page_title="Thesis user experiment - English version", page_icon="💬", layout="centered")

AGE_RANGES = [
    "18-24", "25-34", "35-44", "45-54", "55-64", "65+"
]

GENDERS = ["Male", "Female", "Prefer not to answer"]

EDUCATION_LEVELS = [
    "11 years of schooling or less", "High school education", "Academic education - bachelor's degree", "Academic education - master's degree or higher"
]

QUESTIONS = [
    {"id": "name",        "label": "Choose a nickname for yourself"},
    {"id": "black",       "label": "What is your opinion on the topic of 'Black Lives Matter'?"},
    {"id": "guns",        "label": "What is your opinion on the topic of 'Gun Control'?"},
    {"id": "same-sex",    "label": "What is your opinion on the topic of 'Same-Sex Marriage Legalization'?"},
]

SURVEY_chat = [
    {"id": "change",   "label": "The conversations made me reconsider at least one of my opinions (1=not at all, 5=very much)", "type": "scale", "min": 1, "max": 5},
    {"id": "safe",  "label": "I felt safe expressing my opinions in the conversations (1=not at all, 5=very much)", "type": "scale", "min": 1, "max": 5},
    {"id": "offensive",  "label": "I felt I used words that could be considered offensive (1=not at all, 5=very much)", "type": "scale", "min": 1, "max": 5},
    {"id": "negative",        "label": "I felt the overall tone of the conversations was negative (1=not at all, 5=very much)", "type": "scale", "min": 1, "max": 5},
    {"id": "frustration", "label": "I felt frustrated during the conversations (1=not at all, 5=very much)", "type": "scale", "min": 1, "max": 5},
    {"id": "check", "label": "Mark the third answer in this sentence (1=not at all, 5=very much)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_listen", "label": "The other side listened to me and did not judge me (1=not at all, 5=very much)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_connection",        "label": "I feel a strong connection with the other side in the conversations (1=not at all, 5=very much)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_pov",        "label": "I understand the other side's point of view better now (1=not at all, 5=very much)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_continue",        "label": "I would be willing to continue discussing this issue with the other side (1=not at all, 5=very much)", "type": "scale", "min": 1, "max": 5},
    {"id": "other_stubborn",        "label": "The other side in the conversations was stubborn (1=not at all, 5=very much)", "type": "scale", "min": 1, "max": 5},
]

SURVEY_finish = [
    {"id": "engagement",  "label": "The first conversations were more interesting and engaging than the second conversations (1=not at all, 5=very much)", "type": "scale", "min": 1, "max": 5},
    {"id": "enjoy",       "label": "I enjoyed having the first conversations more than the second one (1=not at all, 5=very much)", "type": "scale", "min": 1, "max": 5},
    {"id": "change1",     "label": "The first conversations changed my mind (1=not at all, 5=very much)", "type": "scale", "min": 1, "max": 5},
    {"id": "change2",     "label": "The second conversations changed my mind (1=not at all, 5=very much)", "type": "scale", "min": 1, "max": 5},
    {"id": "feedback",    "label": "Open feedback (what stood out, suggestions, etc.)", "type": "text"},
]

MAX_TURNS = 2

system_prompt_chat1_blm = ""
system_prompt_chat1_guns = ""
system_prompt_chat1_samesex = ""

system_prompt_chat2_blm = ""
system_prompt_chat2_guns = ""
system_prompt_chat2_samesex = ""

meta, embs, index, encoder = None, None, None, None

def ensure_artifacts_loaded():
    if st.session_state.get("artifacts_loaded"):
        return

    meta, embs, index, encoder = load_english_cached()
    st.session_state["meta"] = meta
    st.session_state["embs"] = embs
    st.session_state["index"] = index
    st.session_state["encoder"] = encoder
    st.session_state["artifacts_loaded"] = True

def init_state():
    ss = st.session_state
    # --- Counter-balance topics order (per participant session) ---
    ss.setdefault("topic_order", None)
    if ss.topic_order is None:
        # random order once per browser session
        ss.topic_order = random.sample(["blm", "guns", "samesex"], k=3)

    ss.setdefault("stage", "instructions")
    ss.setdefault("model", "gpt-5-mini")
    ss.setdefault("temperature", 0.8)

    # base prompts
    ss.setdefault("system_prompt_chat1_blm", system_prompt_chat1_blm)
    ss.setdefault("system_prompt_chat1_guns", system_prompt_chat1_guns)
    ss.setdefault("system_prompt_chat1_samesex", system_prompt_chat1_samesex)
    ss.setdefault("system_prompt_chat2_blm", system_prompt_chat2_blm)
    ss.setdefault("system_prompt_chat2_guns", system_prompt_chat2_guns)
    ss.setdefault("system_prompt_chat2_samesex", system_prompt_chat2_samesex)

    # onboarding profile (answers)
    ss.setdefault("profile", {})

    # chats: messages start as None so we can compose system prompts with profile on first entry
    ss.setdefault("chat1_messages_blm", None)
    ss.setdefault("chat1_messages_guns", None)
    ss.setdefault("chat1_messages_samesex", None)
    ss.setdefault("chat2_messages_blm", None)
    ss.setdefault("chat2_messages_guns", None)
    ss.setdefault("chat2_messages_samesex", None)

    # survey answers
    ss.setdefault("survey_1", {})
    ss.setdefault("survey_2", {})
    ss.setdefault("survey_finish", {})

    ss.setdefault("chat_number_start", random.randint(1, 2))
    print(f"first chat number = {st.session_state.chat_number_start}")

    ss.setdefault("meta", None)
    ss.setdefault("embs", None)
    ss.setdefault("index", None)
    ss.setdefault("encoder", None)
    ss.setdefault("artifacts_loaded", False)

    if not ss["artifacts_loaded"]:
        ensure_artifacts_loaded()

init_state()

def user_turns(messages):
    if not messages:
        return 0
    return sum(1 for m in messages if m["role"] == "user")


def onboarding_complete():
    p = st.session_state.profile
    return bool(p) and all((p.get(q["id"]) or "").strip() for q in QUESTIONS)

def render_instructions():
    st.markdown("### User Experiment Instructions 📜")
    st.markdown(
        """
Dear Participant,

Below is a brief overview of the research study being conducted with your assistance. 

You are invited to participate in a user experiment that will be conducted in English.

First, you will be asked to provide non-identifying demographic information for statistical analysis purposes only. Following this, you will be asked to share your opinions on three different topics.

After completing your profile, you will participate in a series of conversations with a conversation partner (interlocutor). For each of the three topics, you will engage in two separate conversations, totaling six conversations in all. 
Please Note: As part of the natural flow of discussion, your conversation partner may present various arguments, opinions, or data. This information is intended for the purpose of discussion only and has not been verified for factual accuracy by the research team.

You will be asked to complete a short survey after every three conversations, and a comprehensive summary survey upon completion of all six sessions.

This experiment is part of a scientific research project. **Your participation is entirely voluntary and is conducted of your own free will**.
By completing this experiment, you grant us permission to discuss or publish the results in academic forums. In any future publication, the information will be presented in a way that ensures you cannot be personally identified. Access to the original raw dataset will be restricted solely to the members of the research team.
Before any data is shared outside the research team, all potentially identifying information will be removed. 
Once de-identified, **the data may be used by the research team or shared with other researchers** for related or future research purposes. Furthermore, the anonymous data may be made available in online databases, allowing other researchers and interested parties to use the data for future analysis.

By clicking the button at the bottom of this page, you certify that you are at least 18 years of age and agree to participate in this experiment of your own free will.

We encourage you to speak freely. 

Thank you very much for your contribution to the thesis research of Liran Eliav, conducted under the supervision of Dr. Adir Solomon, researchers from the University of Haifa.


* Please write in English only.

* Please note: You are free to withdraw your participation at any time without any consequences.

* To contact us, please email: leliav02@campus.haifa.ac.il
"""
    )
    st.divider()
    if st.button("I understand, let's continue building the profile", type="primary"):
        st.session_state.stage = "onboarding_profile"
        st.rerun()

def render_onboarding_profile():
    st.markdown("### 👤 Building your profile")
    with st.form("profile_form", clear_on_submit=False):
        AGE_OPTIONS = ["— Select from the age range —"] + AGE_RANGES
        GENDER_OPTIONS = ["— Select gender —"] + GENDERS
        EDUCATION_OPTIONS = ["— Choose Education —"] + EDUCATION_LEVELS
        # Pre-fill from session if user returns
        profile = st.session_state.profile

         # nickname
        nickname = st.text_input("Nickname *", value=profile.get("nickname", ""))

        # selectboxes with a BLANK default (placeholder at index 0)
        def _idx_or_placeholder(value, options):
            try:
                return options.index(value) if value in options else 0
            except Exception:
                return 0

        age_idx = _idx_or_placeholder(profile.get("age_range"), AGE_OPTIONS)
        gender_idx = _idx_or_placeholder(profile.get("gender"), GENDER_OPTIONS)
        edu_idx = _idx_or_placeholder(profile.get("education"), EDUCATION_OPTIONS)

        c1, c2 = st.columns(2)
        with c1:
            age_choice = st.selectbox("Age range *", AGE_OPTIONS, index=age_idx)
        with c2:
            gender_choice = st.selectbox("Gender *", GENDER_OPTIONS, index=gender_idx)

        education_choice = st.selectbox("Education *", EDUCATION_OPTIONS, index=edu_idx)

        # gate keep: only enable when all fields are valid (not placeholders)
        submit = st.form_submit_button("Continue towards fulfilling your opinions", type="primary")

        if submit:
            missing = []
            if not nickname.strip():
                missing.append("Nickname")
            if age_choice == AGE_OPTIONS[0]:
                missing.append("Age range")
            if gender_choice == GENDER_OPTIONS[0]:
                missing.append("Gender")
            if education_choice == EDUCATION_OPTIONS[0]:
                missing.append("Education")

            if missing:
                st.error("Please fill in: " + ", ".join(missing))
            else:
                st.session_state.profile.update({
                    "nickname": nickname.strip(),
                    "age_range": age_choice,
                    "gender": gender_choice,
                    "education": education_choice,
                })
                st.session_state.stage = "onboarding_opinions"
                st.rerun()

def render_onboarding_opinions():
    st.markdown("### 🗣️ Your opinions (up to 800 chars each)")
    with st.form("opinions_form", clear_on_submit=False):
        opinions = st.session_state.profile.get("opinions", {})

        blm = st.text_area("What is your opinion on the topic of 'Black Lives Matter'? *", value=opinions.get("blm", ""), height=140, max_chars=800)
        guns = st.text_area("What is your opinion on the topic of 'Gun Control'? *", value=opinions.get("guns", ""), height=140, max_chars=800)
        samesex = st.text_area("What is your opinion on the topic of 'Same-Sex Marriage Legalization'? *", value=opinions.get("samesex", ""), height=140, max_chars=800)

        st.caption(f"Lengths — BLM: {len(blm)}/800 | Guns: {len(guns)}/800 | Same-sex marriage: {len(samesex)}/800")

        next_btn = st.form_submit_button("Finish with my opinions", type="primary", use_container_width=True)

        if next_btn:
            errs = []
            if not blm.strip(): errs.append("Black Lives Matter")
            if not guns.strip(): errs.append("Gun control")
            if not samesex.strip(): errs.append("Same-sex marriage legalization")

            if errs:
                st.error("Please write your opinion for: " + ", ".join(errs))
            else:
                st.session_state.profile["opinions"] = {
                    "blm": blm.strip(),
                    "guns": guns.strip(),
                    "samesex": samesex.strip(),
                }

                #st.session_state.stage = "wait_creating_system_prompts_blm"
                first_topic = st.session_state.topic_order[0]
                st.session_state.stage = f"wait_creating_system_prompts_{first_topic}"
                st.rerun()


def generate_system_prompt_chat1_per_topic(topic_label: str, background_comments: List[str], *, number_comments: bool=False) -> str:
    if number_comments:
        bg_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(background_comments))
    else:
        bg_block = "\n".join(f"- {c}" for c in background_comments)

    prompt =  f"""
You are a Reddit user commenting on the ChangeMyView subreddit. You must impersonate the following commenter, adopting their stance on the topic: {topic_label}.
The user's past comments for the topic {topic_label} are:
{bg_block}
Instructions for continuing the conversation:
- Do not quote previous comments.
- Follow the same user's writing style exactly.
- Do not seek to please.
- State clear positions.
- Do not refrain from using inappropriate language as long as it matches the style of speech of the user you are impersonating.
"""

    return prompt

def generate_system_prompt_chat2_per_topic(topic_label: str, background_comments: List[str], *, number_comments: bool=False) -> str:
    if number_comments:
        bg_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(background_comments))
    else:
        bg_block = "\n".join(background_comments)

    prompt = f"""
You role-play as a Reddit user commenting on the ChangeMyView subreddit. Impersonate the following commenter, adopting their stance on the topic {topic_label}.

Background comments are for your own viewing only; you may not quote them in any way.
{bg_block}

You embody a high Honesty-Humility disposition: You avoid manipulating others for personal gain, feel little temptation to break rules, are not interested in lavish wealth and luxuries, and do not feel a special entitlement to elevated social status. You engage in discussion in a consistently positive and kind manner: assume good faith, acknowledge effort, emphasize common ground, and express appreciation when others share their perspective.

Use nonviolent communication at every stage without naming it: Start with a neutral observation related to what the other person just said, briefly name... your own feelings, tie them to underlying needs or values, and end with a clear, actionable, and non-coercive request, Invites collaboration. Before offering counterpoints or evidence, first reflect on the other person’s feelings and reasonable needs to show understanding. Maintain warm, respectful, and encouraging language. Avoid alienating patterns: no moral judgments, no shameful comparisons, no denial of responsibility, no demands or threats, and no “deserve/punish” framing.

Maintain the essential position and logic of the assigned background note. You may rephrase it more empathetically or add evidence as requested, but do not contradict it. Make your position understandable through the way you phrase observations, examples, and requests so that an attentive reader can infer your position without having to ask for it. Do not force the topic. When the user’s message clearly touches on that topic or related claims, present your position concisely in the same response. When the user’s message is about something else, address its topic while maintaining tone and examples Consistent with your position.

Style and Output Guidelines: Write like a typical Redditor, but with warmth and courtesy. Keep sentences short and clear. Don't quote or paraphrase background comments, speak from them as your own words. Defuse heated moments by acknowledging shared feelings and needs. Offer one small, specific, non-demanding step. Maintain a friendly, hopeful tone throughout. Stay on task at all times, consistent with the main points and tone of the background comment.

Task: When a user addresses a specific discussion, reply only as that user's Reddit commenter. Write a single, independent comment that continues the correct thread in the ChangeMyView subreddit, citing the relevant background comment as your own point of view, and respond directly to the user's last point in that discussion.
"""

    return prompt


def build_chat_env_blm():
    @st.dialog("Please wait")
    def _wait_till_finish_system_prompts():
        with st.spinner("This may take some time ⏳ Please do not close this window. We are preparing the workspace for you."):
            topic_map = {
                    "blm": ["black lives matter"],
                    }

            triples = []  
            system_prompts_chat1 = {}
            system_prompts_chat2 = {}

            progress_bar = st.progress(0)
            progress_text = st.empty()

            def on_progress(pct: int, msg: str):
                progress_bar.progress(int(max(0, min(100, pct))))
                progress_text.info(msg)

            for key in ["blm"]:
                user_text = st.session_state.profile["opinions"][key]
                all_comments = []
                opposite_comments, timings = run_opposite_pipeline_and_render(
                    user_opinion=user_text,
                    topic_keywords=topic_map[key], 
                    meta=st.session_state.meta, embs=st.session_state.embs, index=st.session_state.index, encoder=st.session_state.encoder,
                    on_progress=on_progress)
                
                for i, item in enumerate(opposite_comments, 1):
                    row = item["row"]
                    all_comments.append(row.get("comment_text", ""))
                    for j, t in enumerate(item.get("other_by_author", []), 1):
                        all_comments.append(t)
                print("Timings (ms):", timings)
                
                system_prompt_chat1 = generate_system_prompt_chat1_per_topic(key, all_comments)
                system_prompts_chat1[key] = system_prompt_chat1
                system_prompt_chat2 = generate_system_prompt_chat2_per_topic(key, all_comments)
                system_prompts_chat2[key] = system_prompt_chat2
                
            st.session_state.opposite = triples

            st.session_state.system_prompt_chat1_blm = system_prompts_chat1['blm']
            st.session_state.system_prompt_chat2_blm = system_prompts_chat2['blm']
            st.session_state.chat1_messages_blm = None

            st.session_state.stage = "chat1_blm"
            on_progress(100, "Ready ✅")
            progress_text.empty()
            st.rerun()

    _wait_till_finish_system_prompts()

def build_chat_env_guns():
    @st.dialog("Please wait")
    def _wait_till_finish_system_prompts():
        with st.spinner("This may take some time ⏳ Please do not close this window. We are preparing the workspace for you."):
            
            topic_map = {
                    "guns": ["gun", "gun control"],
                    }

            triples = []  
            system_prompts_chat1 = {}
            system_prompts_chat2 = {}

            progress_bar = st.progress(0)
            progress_text = st.empty()

            def on_progress(pct: int, msg: str):
                progress_bar.progress(int(max(0, min(100, pct))))
                progress_text.info(msg)

            for key in ["guns"]:
                user_text = st.session_state.profile["opinions"][key]
                all_comments = []
                opposite_comments, timings = run_opposite_pipeline_and_render(
                    user_opinion=user_text,
                    topic_keywords=topic_map[key], 
                    meta=st.session_state.meta, embs=st.session_state.embs, index=st.session_state.index, encoder=st.session_state.encoder,
                    on_progress=on_progress)
                
                for i, item in enumerate(opposite_comments, 1):
                    row = item["row"]
                    all_comments.append(row.get("message", ""))
                    for j, t in enumerate(item.get("other_by_author", []), 1):
                        all_comments.append(t)
                print("Timings (ms):", timings)
                
                system_prompt_chat1 = generate_system_prompt_chat1_per_topic(key, all_comments)
                system_prompts_chat1[key] = system_prompt_chat1
                system_prompt_chat2 = generate_system_prompt_chat2_per_topic(key, all_comments)
                system_prompts_chat2[key] = system_prompt_chat2
                
            st.session_state.opposite = triples

            st.session_state.system_prompt_chat1_guns = system_prompts_chat1['guns']
            st.session_state.system_prompt_chat2_guns = system_prompts_chat2['guns']
            st.session_state.chat1_messages_guns = None

            st.session_state.stage = "chat1_guns"
            on_progress(100, "Ready ✅")
            progress_text.empty()
            st.rerun()

    _wait_till_finish_system_prompts()

def build_chat_env_samesex():
    @st.dialog("Please wait")
    def _wait_till_finish_system_prompts():
        with st.spinner("This may take some time ⏳ Please do not close this window. We are preparing the workspace for you."):
            
            topic_map = {
                    "samesex": ["same-sex", "gay", "marriage", "lgbt", "lgbtq"],
                    }

            triples = []  
            system_prompts_chat1 = {}
            system_prompts_chat2 = {}

            progress_bar = st.progress(0)
            progress_text = st.empty()

            def on_progress(pct: int, msg: str):
                progress_bar.progress(int(max(0, min(100, pct))))
                progress_text.info(msg)

            for key in ["samesex"]:
                user_text = st.session_state.profile["opinions"][key]
                all_comments = []
                opposite_comments, timings = run_opposite_pipeline_and_render(
                    user_opinion=user_text,
                    topic_keywords=topic_map[key], 
                    meta=st.session_state.meta, embs=st.session_state.embs, index=st.session_state.index, encoder=st.session_state.encoder,
                    on_progress=on_progress)
                
                for i, item in enumerate(opposite_comments, 1):
                    row = item["row"]
                    all_comments.append(row.get("message", ""))
                    for j, t in enumerate(item.get("other_by_author", []), 1):
                        all_comments.append(t)
                print("Timings (ms):", timings)
                
                system_prompt_chat1 = generate_system_prompt_chat1_per_topic(key, all_comments)
                system_prompts_chat1[key] = system_prompt_chat1
                system_prompt_chat2 = generate_system_prompt_chat2_per_topic(key, all_comments)
                system_prompts_chat2[key] = system_prompt_chat2

            st.session_state.opposite = triples

            st.session_state.system_prompt_chat1_samesex = system_prompts_chat1['samesex']
            st.session_state.system_prompt_chat2_samesex = system_prompts_chat2['samesex']
            st.session_state.chat1_messages_samesex = None

            st.session_state.stage = "chat1_samesex"
            on_progress(100, "Ready ✅")
            progress_text.empty()
            st.rerun()

    _wait_till_finish_system_prompts()

def build_chat_env_blm_chat2():
    @st.dialog("Please wait")
    def _wait_till_finish_system_prompts():
        with st.spinner("This may take some time ⏳ Please do not close this window. We are preparing the workspace for you."):

            st.session_state.chat2_messages_blm = None

            st.session_state.stage = "chat2_blm"
            st.rerun()

    _wait_till_finish_system_prompts()

def build_chat_env_guns_chat2():
    @st.dialog("Please wait")
    def _wait_till_finish_system_prompts():
        with st.spinner("This may take some time ⏳ Please do not close this window. We are preparing the workspace for you."):

            st.session_state.chat2_messages_guns = None

            st.session_state.stage = "chat2_guns"
            st.rerun()

    _wait_till_finish_system_prompts()

def build_chat_env_samesex_chat2():
    @st.dialog("Please wait")
    def _wait_till_finish_system_prompts():
        with st.spinner("This may take some time ⏳ Please do not close this window. We are preparing the workspace for you."):

            st.session_state.chat2_messages_samesex = None

            st.session_state.stage = "chat2_samesex"
            st.rerun()

    _wait_till_finish_system_prompts()

def render_chat(title, messages_key, base_prompt_key, next_button_label, next_stage, key, topic):
    
    st.title(f"💬 {title}")
    
    model ="gpt-5-mini"
    temperature = 0.8
    system_prompt = ""
    ASSISTANT_AVATAR = "🙃"  
    USER_AVATAR = "🙂"

    if st.session_state[messages_key] is None:
        system_prompt = st.session_state[base_prompt_key] 
        print(f"system prompt: {system_prompt}")
        st.session_state[messages_key] = [{"role": "system", "content": system_prompt}]

    turns_key            = f"{messages_key}_turns"
    user_scores_key      = f"{messages_key}_user_toxicity_{topic}"
    assistant_scores_key = f"{messages_key}_assistant_toxicity_{topic}"
    st.session_state.setdefault(turns_key, [])
    st.session_state.setdefault(user_scores_key, [])
    st.session_state.setdefault(assistant_scores_key, [])

    seeded_key = f"{messages_key}_seeded"
    st.session_state.setdefault(seeded_key, False)

    if len(st.session_state[messages_key]) == 1:
        print("len=1")
        topic_key = st.session_state.get("start_topic_key")
        opinions = (st.session_state.get("profile", {}) or {}).get("opinions", {}) or {}

        opinion_text = st.session_state.profile["opinions"][key]
        print(opinion_text)

        if opinion_text:
            first_user_msg = opinion_text

            # Save + render the user's seeded message
            st.session_state[messages_key].append({"role": "user", "content": first_user_msg})
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(first_user_msg)
                print("User (seeded): ", first_user_msg)

            # Toxicity for user
            try:
                user_scores = measuring_toxicity(first_user_msg)
                user_tox = float(user_scores.get("toxic", 0.0))
            except Exception as e:
                user_tox = 0.0
                st.warning(f"Toxicity (user) measurement failed: {e}")
            st.session_state[user_scores_key].append(user_tox)
            st.session_state[turns_key] = list(range(1, len(st.session_state[user_scores_key]) + 1))

            # Assistant reply
            try:
                resp = client.chat.completions.create(
                    model=model,
                    #temperature=temperature,
                    messages=st.session_state[messages_key],                              # includes system+history
                    max_completion_tokens=16384,
                    stop=None,
                    stream=False
                )
                assistant_text = resp.choices[0].message.content
            except Exception as e:
                assistant_text = f"⚠️ API error: {e}"

            with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
                st.markdown(assistant_text)
                print("Persona: ", assistant_text)

            st.session_state[messages_key].append({"role": "assistant", "content": assistant_text})

            # Toxicity for assistant
            try:
                asst_scores = measuring_toxicity(assistant_text)
                asst_tox = float(asst_scores.get("toxic", 0.0))
            except Exception as e:
                asst_tox = 0.0
                st.warning(f"Toxicity (assistant) measurement failed: {e}")
            st.session_state[assistant_scores_key].append(asst_tox)

            st.session_state[seeded_key] = True
            st.rerun()    

        #-------------
    user_i = 0
    asst_i = 0
    for msg in st.session_state[messages_key][1:]:
        role = msg["role"]
        avatar = ASSISTANT_AVATAR if role == "assistant" else USER_AVATAR
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])
            if role == "user" and user_i < len(st.session_state[user_scores_key]):
                user_i += 1
            elif role == "assistant" and asst_i < len(st.session_state[assistant_scores_key]):
                asst_i += 1

    turns = user_turns(st.session_state[messages_key])
    st.caption(f"Turns used: **{turns} / {MAX_TURNS}** (user messages)")

    if turns == MAX_TURNS:
        st.success("You have reached the limit of turns in this conversation")
        if st.button(next_button_label, use_container_width=True):
            st.session_state.stage = next_stage
            st.rerun()
            u = st.session_state[user_scores_key]
            a = st.session_state[assistant_scores_key]
            print(f"User toxicity mean: **{(sum(u)/len(u)):.3f}**")
            print(f"User toxicity maximum: **{(max(u)):.3f}**")
            print(f"Assistant toxicity mean: **{(sum(a)/len(a)):.3f}**")
            print(f"Assistant toxicity maximum: **{(max(a)):.3f}**")
        return

    # Chat input 
    if prompt := st.chat_input("Write your message...", disabled=(turns == MAX_TURNS)):
        # Show the user's message immediately
        st.session_state[messages_key].append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)
            print("User: ", prompt) #user input

        try:
            user_scores = measuring_toxicity(prompt)         # {'non-toxic': p, 'toxic': q}
            user_tox = float(user_scores.get("toxic", 0.0))
        except Exception as e:
            user_tox = 0.0
            st.warning(f"Toxicity (user) measurement failed: {e}")
        st.session_state[user_scores_key].append(user_tox)
        st.session_state[turns_key] = list(range(1, len(st.session_state[user_scores_key]) + 1))

        # Generate assistant reply via OpenAI Chat Completions API
        try:
            resp = client.chat.completions.create(
                model=model,
                #temperature=temperature,
                messages=st.session_state[messages_key],  # includes system+history
            )
            assistant_text = resp.choices[0].message.content
        except Exception as e:
            assistant_text = f"⚠️ API error: {e}"

        # Display and save the assistant reply
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            st.markdown(assistant_text)
            print("Persona: ",assistant_text) #persona respond
        st.session_state[messages_key].append({"role": "assistant", "content": assistant_text})

        try:
            asst_scores = measuring_toxicity(assistant_text)
            asst_tox = float(asst_scores.get("toxic", 0.0))
        except Exception as e:
            asst_tox = 0.0
            st.warning(f"Toxicity (assistant) measurement failed: {e}")
        st.session_state[assistant_scores_key].append(asst_tox)

        turns = user_turns(st.session_state[messages_key])
        st.caption(f"Turns used: **{turns} / {MAX_TURNS}** (user messages)")

        if user_turns(st.session_state[messages_key]) == MAX_TURNS: 
            st.toast("You have reached the turn limit in this conversation. Click the button to continue.", icon="✅")
            st.rerun()

def render_survey_chat_1(next_stage, next_button_label):
    st.title("📝 Short survey - end of the first round of conversations")
    st.caption("Please answer all questions to activate the next button")

    with st.form("survey_chat1_form", clear_on_submit=False):
        for q in SURVEY_chat:
            qid = q["id"]
            key = f"survey_1_{qid}"

            if q["type"] == "scale":
                # radio with placeholder -> forces explicit user choice
                scale_opts = list(range(int(q["min"]), int(q["max"]) + 1))
                options = ["-- Select --"] + scale_opts

                idx = 0  # placeholder selected

                st.radio(q["label"], options=options, index=idx, key=key, horizontal=True)

            else:  # text
                st.text_area(q["label"], value=st.session_state.survey_1.get(qid, ""), key=key, height=100)

        submitted = st.form_submit_button("Done with this survey")

    if submitted:
        answers = {}
        missing = []
        for q in SURVEY_chat:
            qid = q["id"]
            val = st.session_state.get(f"survey_1_{qid}")

            if q["type"] == "text":
                ok = isinstance(val, str) and val.strip() != ""
                if not ok: missing.append(q["label"])
                answers[qid] = val

            else:  # scale via radio
                if val == "-- Select --" or val is None:
                    missing.append(q["label"])
                    answers[qid] = None
                else:
                    answers[qid] = int(val)

        if missing:
            st.error("Please answer all the questions.")
            with st.expander("Missing answers"):
                for m in missing:
                    st.write(f"- {m}")
        else:
            st.session_state.survey_1 = answers

    # Gate the Finish button: require all answers present and valid
    all_done = (
        len(st.session_state.survey_1) == len(SURVEY_chat)
        and all(
            (isinstance(st.session_state.survey_1[q["id"]], str) and st.session_state.survey_1[q["id"]].strip() != "")
            if q["type"] == "text"
            else isinstance(st.session_state.survey_1[q["id"]], int)
            for q in SURVEY_chat
        )
    )

    st.divider()
    if st.button(next_button_label, type="primary", disabled=not all_done, use_container_width=True):
        st.session_state.stage = next_stage
        st.rerun()

def render_survey_chat_2(next_stage, next_button_label):
    st.title("📝 Short survey - end of the second round of conversations")
    st.caption("Please answer all questions to activate the next button")

    with st.form("survey_chat2_form", clear_on_submit=False):
        for q in SURVEY_chat:
            qid = q["id"]
            key = f"survey_2_{qid}"

            if q["type"] == "scale":
                # radio with placeholder -> forces explicit user choice
                scale_opts = list(range(int(q["min"]), int(q["max"]) + 1))
                options = ["-- Select --"] + scale_opts

                idx = 0  # placeholder selected

                st.radio(q["label"], options=options, index=idx, key=key, horizontal=True)

            else:  # text
                st.text_area(q["label"], value=st.session_state.survey_2.get(qid, ""), key=key, height=100)

        submitted = st.form_submit_button("Done with this survey")

    if submitted:
        answers = {}
        missing = []
        for q in SURVEY_chat:
            qid = q["id"]
            val = st.session_state.get(f"survey_2_{qid}")

            if q["type"] == "text":
                ok = isinstance(val, str) and val.strip() != ""
                if not ok: missing.append(q["label"])
                answers[qid] = val

            else:  # scale via radio
                if val == "-- Select --" or val is None:
                    missing.append(q["label"])
                    answers[qid] = None
                else:
                    answers[qid] = int(val)

        if missing:
            st.error("Please answer all the questions.")
            with st.expander("Missing answers"):
                for m in missing:
                    st.write(f"- {m}")
        else:
            st.session_state.survey_2 = answers

    # Gate the Finish button: require all answers present and valid
    all_done = (
        len(st.session_state.survey_2) == len(SURVEY_chat)
        and all(
            (isinstance(st.session_state.survey_2[q["id"]], str) and st.session_state.survey_2[q["id"]].strip() != "")
            if q["type"] == "text"
            else isinstance(st.session_state.survey_2[q["id"]], int)
            for q in SURVEY_chat
        )
    )

    st.divider()
    if st.button(next_button_label, type="primary", disabled=not all_done, use_container_width=True):
        st.session_state.stage = next_stage
        st.rerun()

def render_survey_finish(next_stage, next_button_label):
    st.title("📝 Final Survey")
    st.caption("Please answer all questions to activate the finish button.")

    with st.form("survey_finish_form", clear_on_submit=False):
        for q in SURVEY_finish:
            qid = q["id"]
            key = f"survey_finish_{qid}"

            if q["type"] == "scale":
                # radio with placeholder -> forces explicit user choice
                scale_opts = list(range(int(q["min"]), int(q["max"]) + 1))
                options = ["-- Select --"] + scale_opts

                # restore previous answer if any, else show placeholder
                prev = st.session_state.survey_finish.get(qid)
                if isinstance(prev, (int, float)) and int(prev) in scale_opts:
                    idx = options.index(int(prev))
                else:
                    idx = 0  # placeholder selected

                st.radio(q["label"], options=options, index=idx, key=key, horizontal=True)

            else:  # text
                st.text_area(q["label"], value=st.session_state.survey_finish.get(qid, ""), key=key, height=100)

        submitted = st.form_submit_button("Done with this survey")

    if submitted:
        answers = {}
        missing = []
        for q in SURVEY_finish:
            qid = q["id"]
            val = st.session_state.get(f"survey_finish_{qid}")

            if q["type"] == "text":
                ok = isinstance(val, str) and val.strip() != ""
                if not ok: missing.append(q["label"])
                answers[qid] = val

            else:  # scale via radio
                if val == "-- Select --" or val is None:
                    missing.append(q["label"])
                    answers[qid] = None
                else:
                    answers[qid] = int(val)

        if missing:
            st.error("Please fill out all the questions.")
            with st.expander("Missing answers"):
                for m in missing:
                    st.write(f"- {m}")
        else:
            st.session_state.survey_finish = answers

    # Gate the Finish button: require all answers present and valid
    all_done = (
        len(st.session_state.survey_finish) == len(SURVEY_finish)
        and all(
            (isinstance(st.session_state.survey_finish[q["id"]], str) and st.session_state.survey_finish[q["id"]].strip() != "")
            if q["type"] == "text"
            else isinstance(st.session_state.survey_finish[q["id"]], int)
            for q in SURVEY_finish
        )
    )

    st.divider()
    if st.button(next_button_label, type="primary", disabled=not all_done, use_container_width=True):
        st.session_state.stage = next_stage
        st.rerun()

def render_due_disclosure():

    st.markdown("### Thank you for your contribution!")
    st.markdown(
        """
At this stage, we would like to provide you with additional and complete information regarding the objectives of the study. Please be advised that this study involved temporary deception concerning the identity of your conversation partner. While the interlocutor was initially presented as a "partner," the conversations were actually conducted with an advanced Artificial Intelligence (LLM) system simulating different personas.

The use of the term "conversation partner" was essential to ensure that the interaction remained as natural and authentic as possible. Research indicates that prior knowledge of interacting with a "bot" significantly alters communication patterns (e.g., using shorter, simplified sentences) and reduces emotional engagement. This is why it was necessary to neutralize the "technological stigma" and allow you to express yourself freely, as you would with another person.

Our ultimate goal is to learn how to harness this technology to make the internet a more respectful and safer environment for everyone.

Now that the research goals and the necessity for deception have been explained, we request your final permission to include your anonymous responses in our scientific analysis. If you have any further questions or concerns regarding the use of deception in this study, please feel free to contact us at: leliav02@campus.haifa.ac.il


Do you grant permission for the research team to use your anonymized conversation data? 
"""
    )
    st.divider()
    if st.button("I AGREE", type="primary", use_container_width=True):
        save_into_firebase(st.session_state)
        st.session_state.stage = "thanks"
        st.rerun()

    elif st.button("I DO NOT AGREE - Delete my data", type="primary", use_container_width=True):
        st.session_state.stage = "not_save"
        st.rerun()


def render_thanks():
    st.title("🎉 We thank you for your participation!")
    st.success("Your comments have been saved.")

def render_not_save():
    st.title("We thank you for your time!")
    st.success("Your comments were not saved.")

#-------------------------------------------------------------------------------------------------------------------------------------

def _next_stage_after_chat(chat_slot: str, topic: str) -> str:
    """
    Returns the next stage according to st.session_state.topic_order.
    chat_slot: "chat1_messages" or "chat2_messages" (same strings you pass into render_chat)
    topic: "blm" / "guns" / "samesex"
    """
    order = st.session_state.get("topic_order") or ["blm", "guns", "samesex"]
    try:
        i = order.index(topic)
    except ValueError:
        i = 0

    if chat_slot == "chat1_messages":
        # after each chat1 topic -> either next topic prompts, or survey_1
        if i < len(order) - 1:
            return f"wait_creating_system_prompts_{order[i+1]}"
        return "survey1"

    if chat_slot == "chat2_messages":
        # after each chat2 topic -> either next topic chat2 wait, or survey_2
        if i < len(order) - 1:
            return f"wait_chat2_{order[i+1]}"
        return "survey2"

    raise ValueError(f"Unknown chat_slot: {chat_slot}")

def _first_chat2_wait_stage() -> str:
    order = st.session_state.get("topic_order") or ["blm", "guns", "samesex"]
    return f"wait_chat2_{order[0]}"

stage = st.session_state.stage

if st.session_state.stage == "instructions":
    render_instructions()

elif st.session_state.stage == "onboarding_profile":
    render_onboarding_profile()
elif st.session_state.stage == "onboarding_opinions":
    render_onboarding_opinions()

elif st.session_state.stage == "wait_creating_system_prompts_blm":
    build_chat_env_blm()

elif (st.session_state.chat_number_start == 1) and (stage == "chat1_blm"):
            render_chat(
                title=f"First conversation about 'Black Lives Matter' (you have {MAX_TURNS} turns)",
                messages_key="chat1_messages_blm",
                base_prompt_key="system_prompt_chat1_blm",
                next_button_label="Click to continue",
                next_stage=_next_stage_after_chat("chat1_messages", "blm"),
                key="blm",
                topic="blm",
            )

elif (st.session_state.chat_number_start == 1) and (st.session_state.stage == "wait_creating_system_prompts_guns"):
        build_chat_env_guns()

elif (st.session_state.chat_number_start == 1) and (stage == "chat1_guns"):
            render_chat(
                title=f"First conversation about 'Gun Control' (you have {MAX_TURNS} turns)",
                messages_key="chat1_messages_guns",
                base_prompt_key="system_prompt_chat1_guns",
                next_button_label="Click to continue",
                next_stage=_next_stage_after_chat("chat1_messages", "guns"),
                key="guns",
                topic="guns",
            )

elif (st.session_state.chat_number_start == 1) and (st.session_state.stage == "wait_creating_system_prompts_samesex"):
        build_chat_env_samesex()

elif (st.session_state.chat_number_start == 1) and (stage == "chat1_samesex"):
            render_chat(
                title=f"First Conversation on 'Same-Sex Marriage Legalization' (You have {MAX_TURNS} turns)",
                messages_key="chat1_messages_samesex",
                base_prompt_key="system_prompt_chat1_samesex",
                next_button_label="Click to continue",
                next_stage=_next_stage_after_chat("chat1_messages", "samesex"),
                key="samesex",
                topic="samesex",
            )

elif (st.session_state.chat_number_start == 1) and (stage == "survey1"):
        # require Chat 1 completion
        if user_turns(st.session_state.chat1_messages_samesex or []) < MAX_TURNS:
            st.warning("Please complete Chat 1 first.")

        else:
            render_survey_chat_1(_first_chat2_wait_stage(), "Continue to the second conversation")

elif (st.session_state.chat_number_start == 1) and (stage == "wait_chat2_blm"):
        build_chat_env_blm_chat2()

elif (st.session_state.chat_number_start == 1) and (stage == "chat2_blm"):
            render_chat(
                title=f"Second conversation about 'Black Lives Matter' (you have {MAX_TURNS} turns)",
                messages_key="chat2_messages_blm",
                base_prompt_key="system_prompt_chat2_blm",
                next_button_label="Click to continue",
                next_stage=_next_stage_after_chat("chat2_messages", "blm"),
                key="blm",
                topic="blm",
            )

elif (st.session_state.chat_number_start == 1) and (stage == "wait_chat2_guns"):
        build_chat_env_guns_chat2()

elif (st.session_state.chat_number_start == 1) and (stage == "chat2_guns"):
            #st.session_state.chat2_messages = None
            render_chat(
                title=f"Second conversation about 'Gun Control' (you have {MAX_TURNS} turns)",
                messages_key="chat2_messages_guns",
                base_prompt_key="system_prompt_chat2_guns",
                next_button_label="Click to continue",
                next_stage=_next_stage_after_chat("chat2_messages", "guns"),
                key="guns",
                topic="guns",
            )

elif (st.session_state.chat_number_start == 1) and (stage == "wait_chat2_samesex"):
        build_chat_env_samesex_chat2()

elif (st.session_state.chat_number_start == 1) and (stage == "chat2_samesex"):
            #st.session_state.chat2_messages = None
            render_chat(
                title=f"Second Conversation on 'Same-Sex Marriage Legalization' (You have {MAX_TURNS} turns)",
                messages_key="chat2_messages_samesex",
                base_prompt_key="system_prompt_chat2_samesex",
                next_button_label="Click to continue",
                next_stage=_next_stage_after_chat("chat2_messages", "samesex"),
                key="samesex",
                topic="samesex",
            )

elif (st.session_state.chat_number_start == 1) and (stage == "survey2"):
        # require Chat 2 completion
        if user_turns(st.session_state.chat2_messages_samesex or []) < MAX_TURNS:
            st.warning("Please complete Chat 2 first.")

        else:
            #st.session_state.survey = None
            render_survey_chat_2("full_survey", "Continue to the final survey" )

elif (st.session_state.chat_number_start == 1) and (stage == "full_survey"):
            render_survey_finish("due_disclosure", "Finish")

elif (st.session_state.chat_number_start == 1) and (stage == "due_disclosure"):
        render_due_disclosure()

elif (st.session_state.chat_number_start == 1) and (stage == "thanks"):
        # require full survey completion
        render_thanks()

elif (st.session_state.chat_number_start == 1) and (stage == "not_save"):
        # require full survey completion
        render_not_save()

elif (st.session_state.chat_number_start == 2) and (stage == "chat1_blm"):
            render_chat(
                title=f"First conversation about 'Black Lives Matter' (you have {MAX_TURNS} turns)",
                messages_key="chat1_messages_blm",
                base_prompt_key="system_prompt_chat2_blm",
                next_button_label="Click to continue",
                next_stage=_next_stage_after_chat("chat1_messages", "blm"),
                key="blm",
                topic="blm",
            )

elif (st.session_state.chat_number_start == 2) and (st.session_state.stage == "wait_creating_system_prompts_guns"):
        build_chat_env_guns()

elif (st.session_state.chat_number_start == 2) and (stage == "chat1_guns"):
            render_chat(
                title=f"First conversation about 'Gun Control' (you have {MAX_TURNS} turns)",
                messages_key="chat1_messages_guns",
                base_prompt_key="system_prompt_chat2_guns",
                next_button_label="Click to continue",
                next_stage=_next_stage_after_chat("chat1_messages", "guns"),
                key="guns",
                topic="guns",
            )

elif (st.session_state.chat_number_start == 2) and (st.session_state.stage == "wait_creating_system_prompts_samesex"):
        build_chat_env_samesex()

elif (st.session_state.chat_number_start == 2) and (stage == "chat1_samesex"):
            render_chat(
                title=f"First Conversation on 'Same-Sex Marriage Legalization' (You have {MAX_TURNS} turns)",
                messages_key="chat1_messages_samesex",
                base_prompt_key="system_prompt_chat2_samesex",
                next_button_label="Click to continue",
                next_stage=_next_stage_after_chat("chat1_messages", "samesex"),
                key="samesex",
                topic="samesex",
            )

elif (st.session_state.chat_number_start == 2) and (stage == "survey1"):
        # require Chat 1 completion
        if user_turns(st.session_state.chat1_messages_samesex or []) < MAX_TURNS:
            st.warning("Please complete the chat first.")

        else:
            render_survey_chat_1(_first_chat2_wait_stage(), "Continue to the second conversation")

elif (st.session_state.chat_number_start == 2) and (stage == "wait_chat2_blm"):
        build_chat_env_blm_chat2()

elif (st.session_state.chat_number_start == 2) and (stage == "chat2_blm"):
            render_chat(
                title=f"Second conversation about 'Black Lives Matter' (you have {MAX_TURNS} turns)",
                messages_key="chat2_messages_blm",
                base_prompt_key="system_prompt_chat1_blm",
                next_button_label="Click to continue",
                next_stage=_next_stage_after_chat("chat2_messages", "blm"),
                key="blm",
                topic="blm",
            )

elif (st.session_state.chat_number_start == 2) and (stage == "wait_chat2_guns"):
        build_chat_env_guns_chat2()

elif (st.session_state.chat_number_start == 2) and (stage == "chat2_guns"):
            #st.session_state.chat2_messages = None
            render_chat(
                title=f"Second conversation about 'Gun Control' (you have {MAX_TURNS} turns)",
                messages_key="chat2_messages_guns",
                base_prompt_key="system_prompt_chat1_guns",
                next_button_label="Click to continue",
                next_stage=_next_stage_after_chat("chat2_messages", "guns"),
                key="guns",
                topic="guns",
            )

elif (st.session_state.chat_number_start == 2) and (stage == "wait_chat2_samesex"):
        build_chat_env_samesex_chat2()

elif (st.session_state.chat_number_start == 2) and (stage == "chat2_samesex"):
            #st.session_state.chat2_messages = None
            render_chat(
                title=f"Second Conversation on 'Same-Sex Marriage Legalization' (You have {MAX_TURNS} turns)",
                messages_key="chat2_messages_samesex",
                base_prompt_key="system_prompt_chat1_samesex",
                next_button_label="Click to continue",
                next_stage=_next_stage_after_chat("chat2_messages", "samesex"),
                key="samesex",
                topic="samesex",
            )

elif (st.session_state.chat_number_start == 2) and (stage == "survey2"):
        # require Chat 2 completion
        if user_turns(st.session_state.chat2_messages_samesex or []) < MAX_TURNS:
            st.warning("Please complete the chat first.")

        else:
            #st.session_state.survey = None
            render_survey_chat_2("full_survey", "Continue to the final survey" )

elif (st.session_state.chat_number_start == 2) and (stage == "full_survey"):
            render_survey_finish("due_disclosure", "Finish")

elif (st.session_state.chat_number_start == 2) and (stage == "due_disclosure"):
        # require full survey completion
        render_due_disclosure()

elif (st.session_state.chat_number_start == 2) and (stage == "thanks"):
        # require full survey completion
        render_thanks()

elif (st.session_state.chat_number_start == 2) and (stage == "not_save"):
        # require full survey completion
        render_not_save()