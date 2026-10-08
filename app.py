import streamlit as st
import re

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #07111f;
        color: #f1f5f9;
    }

    section[data-testid="stSidebar"] {
        background-color: #252833;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #315b8a;
        background: linear-gradient(
            135deg,
            #2563eb,
            #6366f1
        );
        color: white;
        font-weight: 600;
        min-height: 42px;
    }

    .stButton > button:hover {
        border-color: #818cf8;
        color: white;
    }

    .stTextInput input {
        background-color: #0b1728;
        color: white;
        border: 1px solid #315b8a;
        border-radius: 10px;
    }

    .stChatInput > div {
        background-color: #0d1b2e;
        border: 1px solid #315b8a;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "transcript_loaded" not in st.session_state:
    st.session_state.transcript_loaded = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ============================================================
# EXTRACT YOUTUBE VIDEO ID
# ============================================================

def extract_video_id(url_or_id):

    value = url_or_id.strip()

    # Direct video ID
    if re.fullmatch(
        r"[A-Za-z0-9_-]{11}",
        value
    ):
        return value

    # youtube.com/watch?v=VIDEO_ID
    match = re.search(
        r"(?:v=)([A-Za-z0-9_-]{11})",
        value
    )

    if match:
        return match.group(1)

    # youtu.be/VIDEO_ID
    match = re.search(
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        value
    )

    if match:
        return match.group(1)

    # youtube.com/embed/VIDEO_ID
    match = re.search(
        r"youtube\.com/embed/([A-Za-z0-9_-]{11})",
        value
    )

    if match:
        return match.group(1)

    return None


# ============================================================
# LOAD TRANSCRIPT
# ============================================================

def load_transcript(video_id):

    api = YouTubeTranscriptApi()

    # --------------------------------------------------------
    # First try English
    # --------------------------------------------------------

    try:

        transcript_list = api.fetch(
            video_id,
            languages=["en"]
        )

        transcript = " ".join(
            chunk.text
            for chunk in transcript_list
        )

        if transcript:
            return transcript, "English"

    except Exception:
        pass


    # --------------------------------------------------------
    # If English fails, try Hindi
    # --------------------------------------------------------

    try:

        transcript_list = api.fetch(
            video_id,
            languages=["hi"]
        )

        transcript = " ".join(
            chunk.text
            for chunk in transcript_list
        )

        if transcript:
            return transcript, "Hindi"

    except Exception:
        pass


    # --------------------------------------------------------
    # If both fail
    # --------------------------------------------------------

    raise Exception(
        "No English or Hindi transcript was found for this video."
    )


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vector_store(transcript):

    # --------------------------------------------------------
    # STEP 1: Split transcript
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents(
        [transcript]
    )


    # --------------------------------------------------------
    # STEP 2: Multilingual Embeddings
    # --------------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
    )


    # --------------------------------------------------------
    # STEP 3: FAISS
    # --------------------------------------------------------

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store, len(chunks)


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(vector_store, question):

    # --------------------------------------------------------
    # STEP 1: Retriever
    # --------------------------------------------------------

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 8
        }
    )


    # --------------------------------------------------------
    # STEP 2: Retrieve relevant documents
    # --------------------------------------------------------

    documents = retriever.invoke(
        question
    )


    # --------------------------------------------------------
    # STEP 3: Create context
    # --------------------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    # --------------------------------------------------------
    # STEP 4: Prompt
    # --------------------------------------------------------

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful YouTube video assistant.

Your job is to answer the user's question using
the transcript context provided below.

IMPORTANT RULES:

1. Use the context to answer the question.
2. Do not invent information.
3. If the context contains relevant information,
   explain it clearly.
4. The transcript may be in Hindi or English.
5. The user may ask the question in English or Hindi.
6. You can answer in the same language as the user.
7. Only say "I could not find the answer in the video"
   when the context truly does not contain the answer.

Transcript Context:
-------------------
{context}
-------------------

User Question:
{question}

Answer:
"""
    )


    # --------------------------------------------------------
    # STEP 5: Ollama
    # --------------------------------------------------------

    llm = ChatOllama(
        model="llama3.2"
    )


    # --------------------------------------------------------
    # STEP 6: Chain
    # --------------------------------------------------------

    chain = prompt | llm


    # --------------------------------------------------------
    # STEP 7: Generate answer
    # --------------------------------------------------------

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )


    return response.content, documents


# ============================================================
# HEADER
# ============================================================

st.title("🎥 YouTube RAG Assistant")

st.caption(
    "Ask questions about any YouTube video "
    "using RAG + FAISS + Ollama"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🔗 YouTube Video")

    st.write(
        "Paste a YouTube video URL or video ID "
        "and load the video."
    )


    # --------------------------------------------------------
    # URL INPUT
    # --------------------------------------------------------

    video_input = st.text_input(
        "YouTube URL / Video ID",
        placeholder="https://youtube.com/watch?v=..."
    )


    # --------------------------------------------------------
    # LOAD VIDEO BUTTON
    # --------------------------------------------------------

    load_video = st.button(
        "▶ Load Video",
        use_container_width=True
    )


    # ========================================================
    # LOAD VIDEO
    # ========================================================

    if load_video:

        if not video_input:

            st.error(
                "Please enter a YouTube URL or Video ID."
            )

        else:

            video_id = extract_video_id(
                video_input
            )


            if not video_id:

                st.error(
                    "Invalid YouTube URL or Video ID."
                )

            else:

                try:

                    # ------------------------------------------------
                    # LOAD TRANSCRIPT
                    # ------------------------------------------------

                    with st.spinner(
                        "📥 Loading YouTube transcript..."
                    ):

                        transcript, transcript_language = (
                            load_transcript(
                                video_id
                            )
                        )


                    st.success(
                        f"Transcript loaded! "
                        f"Language: {transcript_language}"
                    )


                    # ------------------------------------------------
                    # CREATE VECTOR STORE
                    # ------------------------------------------------

                    with st.spinner(
                        "🧠 Creating embeddings and FAISS index..."
                    ):

                        vector_store, chunk_count = (
                            create_vector_store(
                                transcript
                            )
                        )


                    # ------------------------------------------------
                    # SAVE SESSION STATE
                    # ------------------------------------------------

                    st.session_state.vector_store = (
                        vector_store
                    )

                    st.session_state.video_id = (
                        video_id
                    )

                    st.session_state.transcript_loaded = (
                        True
                    )

                    st.session_state.chunk_count = (
                        chunk_count
                    )

                    st.session_state.chat_history = []


                    st.success(
                        f"Video ready! "
                        f"{chunk_count} chunks created."
                    )


                except Exception as e:

                    st.error(
                        f"Error: {str(e)}"
                    )


    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    if st.session_state.transcript_loaded:

        st.divider()

        st.subheader("📊 Video Information")

        st.write(
            "**Video ID**"
        )

        st.code(
            st.session_state.video_id
        )

        st.write(
            f"**Chunks:** "
            f"{st.session_state.chunk_count}"
        )

        st.write(
            "**Status:** 🟢 Ready"
        )


    # ========================================================
    # QUICK QUESTIONS
    # ========================================================

    st.divider()

    st.subheader("💡 Quick Questions")


    quick_questions = [
        "What is the main topic?",
        "What is Context API?",
        "How does Context API work?",
        "What is localStorage?"
    ]


    for quick_question in quick_questions:

        if st.button(
            quick_question,
            use_container_width=True
        ):

            if not st.session_state.transcript_loaded:

                st.warning(
                    "Please load a video first."
                )

            else:

                # Save user question

                st.session_state.chat_history.append(
                    {
                        "role": "user",
                        "content": quick_question
                    }
                )


                # Generate answer

                with st.spinner(
                    "🤖 Searching the video..."
                ):

                    answer, documents = ask_question(
                        st.session_state.vector_store,
                        quick_question
                    )


                # Save answer

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


                st.rerun()


# ============================================================
# MAIN AREA
# ============================================================

chat_col, info_col = st.columns(
    [3, 1]
)


# ============================================================
# CHAT
# ============================================================

with chat_col:

    st.subheader("💬 Chat")

    st.caption(
        "Ask anything about the loaded YouTube video."
    )


    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    if st.button("🗑️ Clear Chat"):

        st.session_state.chat_history = []

        st.rerun()


    st.divider()


    # ========================================================
    # WELCOME MESSAGE
    # ========================================================

    if not st.session_state.chat_history:

        with st.chat_message("assistant"):

            st.write(
                "Hi! 👋 I'm your YouTube video assistant."
            )

            st.write(
                "Load a YouTube video and ask me "
                "anything about its transcript."
            )


    # ========================================================
    # CHAT HISTORY
    # ========================================================

    for message in st.session_state.chat_history:

        if message["role"] == "user":

            with st.chat_message("user"):

                st.write(
                    message["content"]
                )

        else:

            with st.chat_message("assistant"):

                st.write(
                    message["content"]
                )


    # ========================================================
    # CHAT INPUT
    # ========================================================

    question = st.chat_input(
        "Ask something about the video..."
    )


    if question:

        # ----------------------------------------------------
        # CHECK VIDEO
        # ----------------------------------------------------

        if not st.session_state.transcript_loaded:

            st.warning(
                "Please load a video first."
            )

        else:

            # ------------------------------------------------
            # USER MESSAGE
            # ------------------------------------------------

            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question
                }
            )


            with st.chat_message("user"):

                st.write(
                    question
                )


            # ------------------------------------------------
            # ASSISTANT MESSAGE
            # ------------------------------------------------

            with st.chat_message("assistant"):

                with st.spinner(
                    "🤖 Searching the video and thinking..."
                ):

                    answer, documents = ask_question(
                        st.session_state.vector_store,
                        question
                    )


                st.write(
                    answer
                )


            # ------------------------------------------------
            # SAVE ANSWER
            # ------------------------------------------------

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )


            # ------------------------------------------------
            # DEBUG: RETRIEVED CHUNKS
            # ------------------------------------------------

            with st.expander(
                "🔍 See retrieved transcript chunks"
            ):

                for i, document in enumerate(
                    documents,
                    start=1
                ):

                    st.markdown(
                        f"**Chunk {i}**"
                    )

                    st.write(
                        document.page_content
                    )

                    st.divider()


# ============================================================
# RIGHT SIDE
# ============================================================

with info_col:

    st.subheader("🧠 How it works")

    st.info(
        """
**1. Transcript**

YouTube transcript is loaded.


**2. Chunking**

Transcript is divided into smaller pieces.


**3. Embeddings**

Each chunk becomes a vector.


**4. FAISS**

FAISS finds relevant chunks.


**5. Ollama**

Llama 3.2 reads the relevant context.


**6. Answer**

Llama generates the final answer.
"""
    )


    st.subheader("⚙️ System")

    st.success(
        """
🟢 HuggingFace

🟢 FAISS

🟢 Ollama

🟢 Llama 3.2
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "YouTube RAG Assistant • "
    "Streamlit + LangChain + FAISS + Ollama"
)