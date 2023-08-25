"""
Purpose: Web application for docs2chat.
"""


import os
from pathlib import Path
from PIL import Image
import streamlit as st


from docs2chat.apps.utils import ChainFactory
from docs2chat.config import config


APPS_DIR = Path(os.path.realpath(__file__)).parents[0].absolute()
IMAGES_DIR = APPS_DIR / "images"
HOMEPAGE_INTRODUCTION = """
# Welcom to Docs2Chat

No more endless 'ctrl-f-ing'! Large Language Models (LLMs) are enabling
the next generation of information retrieval toolkits. Docs2Chat
utilizes LLMs to allow you to query your documents in a conversational
format. Simply configure the chat on the left-hand navigation pannel,
hit 'Start Chat' and type a question.

The three chat types can be summarized as follows:

  1. **Search:**
  2. **Snip:**
  3. **Generative:**

"""

# Set the main page config.
st.set_page_config(
    page_title="Docs2Chat",
    page_icon=str(IMAGES_DIR / "robot_reading_ai_generated.png")
)

col1, col2 = st.columns([7,3])

with col1:
    st.write(HOMEPAGE_INTRODUCTION, unsafe_allow_html=True)

with col2:
    st.image(Image.open(IMAGES_DIR / "robot_reading_ai_generated.png"))
    st.image(Image.open(IMAGES_DIR / "icon.png"))

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

with st.sidebar:
        st.subheader("Chat Configuration")
        chain_type = st.radio(
            "Select a chat type:",
            ["search", "snip", "generative"]
        )
        start_chat = st.button("Start Chat")

if start_chat:
    with st.spinner("Initiating chat session. This may take a minute ..."):
        chain, format_func = ChainFactory(
        chain_type=chain_type,
        docs_dir="/home/ubuntu/projects/docs2chat/notebooks/docs",
        config_obj=config,
        num_return_docs=4,
        return_threshold=0
    )
    user_question = st.text_input("Ask a question about your documents:")