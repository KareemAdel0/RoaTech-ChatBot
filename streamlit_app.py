import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import pickle, time, difflib, re
import streamlit as st
import numpy as np
import pandas as pd
from roadmaps import roadmaps
from dotenv import load_dotenv
from googleapiclient.discovery import build

# Load environment variables
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Feedback path
FEEDBACK_FILE = os.path.join("datasets", "feedback.csv")

# ============== Load Models ============== #
with open(os.path.join("model_pkl_files", "embedder.pkl"), "rb") as f:
    embedder = pickle.load(f)
with open(os.path.join("model_pkl_files", "transformer_logestic_model.pkl"), "rb") as f:
    logestic_model = pickle.load(f)
with open(os.path.join("model_pkl_files", "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

# ============== Greeting Detection ============== #
def is_greeting(text):
    greetings = [
        "hello", "hi", "hey", "good morning", "good evening",
        "salam", "السلام عليكم", "مرحبا", "أهلا", "صباح الخير",
        "مساء الخير", "كيف حالك", "أزيك", "عامل اى", "شلونك",
        "how you doing", "my friend", "my dear"
    ]
    return any(greet in text.lower() for greet in greetings)

# ============== Embedding Extraction ============== #
def get_embedding(text):
    return embedder.encode([text.lower().strip()])

# ============== YouTube Video Search ============== #
def search_youtube_video(query):
    try:
        request = youtube.search().list(
            part="id",
            q=query,
            type="video",
            maxResults=1
        )
        response = request.execute()
        if response["items"]:
            video_id = response["items"][0]["id"]["videoId"]
            return f"https://www.youtube.com/watch?v={video_id}"
        return "No video found"
    except Exception as e:
        return f"Error fetching video: {str(e)}"

# ============== Display Roadmap Anim ============== #
def display_roadmap_anim(track, levels_to_show):
    data = roadmaps.get(track, {})
    intro = data.get("intro", "No intro available.")
    levels = data.get("levels", {})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        content = f"### 🟨 {track.title()} Track\n\n{intro}\n\n"
        placeholder.markdown(content)
        time.sleep(1)

        # Counter for steps across all levels
        step_counter = 1

        # Loop through the levels to show (beginner, intermediate, advanced as needed)
        for level in levels_to_show:
            level_data = levels.get(level, {})
            steps = level_data.get("steps", [])
            for step in steps:
                content += f"**Step {step_counter}:** {step.get('step', 'No step defined')}\n"
                for link in step.get("resources", []):
                    content += f"- [{link}]({link})\n"
                content += "\n"
                step_counter += 1
            placeholder.markdown(content)
            time.sleep(1)

        st.session_state.messages.append({"role": "assistant", "content": content})

# ============== Streamlit Setup & Custom CSS ============== #
st.set_page_config(page_title="💬 RoaTech", page_icon="🤖", layout="wide")
st.markdown("""
<style>
/* Chat bubbles */
.chat-user .stMarkdown, .chat-assistant .stMarkdown {
    border-radius: 12px;
    padding: 8px 12px;
    margin-bottom: 6px;
    max-width: 75%;
}
.chat-user .stMarkdown {
    background-color: #4a4e69;
    align-self: flex-end;
}
.chat-assistant .stMarkdown {
    background-color: #22223b;
    align-self: flex-start;
}
/* Input box */
.stTextInput>div>div>input {
    background-color: #f2e9e4;
    color: #22223b;
    border-radius: 8px;
    padding: 10px;
}
/* Buttons */
.stButton>button {
    background-color: #fd6965;
    color: #fff;
    border-radius: 8px;
    padding: 10px 20px;
    margin: 4px;
    width: 100%;  /* Makes button fill the column width */
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #c9ada7;
}
/* Sidebar */
.sidebar .sidebar-content {
    background-color: #2b2d42;
    color: #edf2f4;
    position: fixed;
    top: 0;
    bottom: 0;
    width: 20%;
}
</style>
""", unsafe_allow_html=True)

# ============== Sidebar (Fixed) ============== #
st.sidebar.title("🤖 RoaTech Chatbot")
st.sidebar.markdown("Your interactive guide to tech careers! 🚀")

# Side-by-side buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.clear()
with col2:
    if st.button("🔚 End Chat", key="end_chat"):
        thank_you_message = "شكراً لهذه المحادثة! 😊"
        st.session_state.messages.append({"role": "assistant", "content": thank_you_message})
        st.session_state.show_feedback = True


# ============== Main Chat Area ============== #
if "messages" not in st.session_state:
    st.session_state.messages = []
if "welcomed" not in st.session_state:
    st.session_state.welcomed = False
if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = "initial"
if "selected_track" not in st.session_state:
    st.session_state.selected_track = None
if "show_track_form" not in st.session_state:
    st.session_state.show_track_form = False
if "user_level" not in st.session_state:
    st.session_state.user_level = None
if "chat_ended" not in st.session_state:
    st.session_state.chat_ended = False

if not st.session_state.welcomed:
    welcome_msg = "🎉 Welcome to RoaTech Chatbot!"
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    st.session_state.welcomed = True

for m in st.session_state.messages:
    with st.chat_message(m['role']):
        st.markdown(m['content'])

prompt = st.chat_input("Type your message here...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    emb = get_embedding(prompt)
    state = st.session_state.conversation_state
    response = None
    if state == "initial":
        if is_greeting(prompt):
            response = ("👋 Hello! Please choose:\n\n"
                        "1. Explore all tech tracks\n"
                        "2. Get a roadmap for a specific track")
            st.session_state.conversation_state = "goal_choice"
        else:
            response = "Hello! Could you start with a greeting? 😊"
    elif state == "goal_choice":
        text = prompt.lower()
        cleaned = re.sub(r'^(yes|okay|ok)[\s,]*', '', text).strip()
        if "1" in cleaned or "explore" in cleaned:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                content = "### 🎯 Available Tech Tracks:\n\n"
                placeholder.markdown(content)
                time.sleep(0.5)
                for t in roadmaps:
                    intro = roadmaps[t].get("intro", "")
                    content += f"- **{t.title()}**: {intro[:80]}...\n"
                    placeholder.markdown(content)
                    time.sleep(0.3)
                content += "\nType any track name to get its roadmap."
                placeholder.markdown(content)
            for line in content.split("\n"):
                if line.strip():
                    st.session_state.messages.append({"role": "assistant", "content": line})
            st.session_state.conversation_state = "track_choice"
        elif "2" in cleaned or "roadmap" in cleaned:
            response = "Type the track name you want a roadmap for."
            st.session_state.conversation_state = "track_choice"
        else:
            if cleaned in roadmaps or (match := difflib.get_close_matches(cleaned, roadmaps.keys(), n=1, cutoff=0.6)):
                st.session_state.selected_track = cleaned if cleaned in roadmaps else match[0]
                st.session_state.conversation_state = "level_choice"
            else:
                response = ("❌ I didn't understand.\n"
                            "1. Explore all tech tracks\n"
                            "2. Get a roadmap for a specific track")
    elif state == "track_choice":
        txt = prompt.lower().strip()
        if txt in roadmaps or (match := difflib.get_close_matches(txt, roadmaps.keys(), n=1, cutoff=0.6)):
            tr = txt if txt in roadmaps else match[0]
            st.session_state.selected_track = tr
            response = "What's your level? (beginner/intermediate/advanced)"
            st.session_state.conversation_state = "level_choice"
        else:
            probs = logestic_model.predict_proba(emb)[0]
            idx = np.argmax(probs); conf = probs[idx]
            tr = label_encoder.inverse_transform([idx])[0]
            
            if conf >= 0.5 and tr in roadmaps:  # Lowered threshold to 0.5
                st.session_state.selected_track = tr
                response = "What's your level? (beginner/intermediate/advanced)"
                st.session_state.conversation_state = "level_choice"
            else:
                # Enhanced fuzzy matching for single-word inputs
                if len(txt.split()) == 1:
                    matches = difflib.get_close_matches(txt, [t.lower().split()[0] for t in roadmaps.keys()], n=1, cutoff=0.7)
                    if matches:
                        matched_word = matches[0]
                        for track in roadmaps.keys():
                            if track.lower().startswith(matched_word):
                                tr = track
                                break
                        st.session_state.selected_track = tr
                        response = "What's your level? (beginner/intermediate/advanced)"
                        st.session_state.conversation_state = "level_choice"
                    else:
                        st.session_state.show_track_form = True
                        response = "❌ Couldn't identify track. Please choose from the list."
                else:
                    st.session_state.show_track_form = True
                    response = "❌ Couldn't identify track. Please choose from the list."

    elif state == "level_choice":
        lvl = prompt.lower().strip()
        choices = ["beginner", "intermediate", "advanced"]
        if lvl not in choices:
            m = difflib.get_close_matches(lvl, choices, n=1, cutoff=0.6)
            if m: lvl = m[0]
        if lvl in choices:
            # Determine which levels to show based on user input
            if lvl == "beginner":
                levels_to_show = ["beginner", "intermediate", "advanced"]
            elif lvl == "intermediate":
                levels_to_show = ["intermediate", "advanced"]
            else:  # lvl == "advanced"
                levels_to_show = ["advanced"]

            st.session_state.user_level = lvl
            display_roadmap_anim(st.session_state.selected_track, levels_to_show)
            st.session_state.conversation_state = "track_choice"
        else:
            response = "Please choose: beginner, intermediate, or advanced"
    if response:
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

# ============== Feedback ============== #
if st.session_state.get("show_feedback", False):
    with st.container():
        st.markdown("""
        <div style="padding: 15px; border-radius: 12px; background-color: #2b2d42; color: #ffffff;">
            <h3 style="color:#fcd5ce;">💬 We’d love your feedback!</h3>
            <p style="font-size: 16px;">How would you rate your experience with our chatbot?</p>
        </div>
        """, unsafe_allow_html=True)

        rating = st.slider("⭐ Rate the chat (1 = poor, 5 = excellent)", 1, 5, 3)
        comment = st.text_area("✏️ Share any comments or suggestions")

        submit = st.button("🚀 Submit Feedback")

        if submit:
            df = pd.DataFrame({
                "rating": [rating],
                "comment": [comment],
                "time": [pd.Timestamp.now()]
            })
            df.to_csv(FEEDBACK_FILE, mode="a", index=False,
                      header=not os.path.isfile(FEEDBACK_FILE))
            st.success("✅ Thanks for your feedback! We appreciate it. 🙌")
            st.session_state.show_feedback = False
