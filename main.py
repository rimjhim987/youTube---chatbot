# import os
# from dotenv import load_dotenv
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import FAISS
# # from langchain_core.prompts import PromptTemplate

# load_dotenv()

# api_key = os.getenv("OPENAI_API_KEY")

# print("API key loaded:", bool(api_key))


# video_id = "J5_-l7WIO_w"

# api = YouTubeTranscriptApi()

# transcript_list = api.fetch(video_id, languages=["hi"])

# transcript = " ".join(chunk.text for chunk in transcript_list)

# #print(transcript)


# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# chunks = splitter.create_documents([transcript])

# #print(chunks)
# #print(len(chunks))
# #print(chunks[50].page_content)

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# vector_store = FAISS.from_documents(chunks, embeddings)

# print(vector_store.index_to_docstore_id)

# ==========================================
# IMPORTS
# ==========================================

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama


# ==========================================
# 1. YOUTUBE VIDEO
# ==========================================

video_id = "J5_-l7WIO_w"


# ==========================================
# 2. GET YOUTUBE TRANSCRIPT
# ==========================================

api = YouTubeTranscriptApi()

transcript_list = api.fetch(
    video_id,
    languages=["hi"]
)

transcript = " ".join(
    chunk.text for chunk in transcript_list
)

print("Transcript loaded successfully!")
print("Transcript length:", len(transcript))


# ==========================================
# 3. SPLIT TRANSCRIPT INTO CHUNKS
# ==========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.create_documents(
    [transcript]
)

print("Number of chunks:", len(chunks))


# ==========================================
# 4. FREE LOCAL EMBEDDINGS
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# ==========================================
# 5. CREATE FAISS VECTOR STORE
# ==========================================

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector store created!")


# ==========================================
# 6. CREATE RETRIEVER
# ==========================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4
    }
)

print("Retriever created!")


# ==========================================
# 7. USER QUESTION
# ==========================================

question = "What is the main topic discussed in this video?"


# ==========================================
# 8. RETRIEVE RELEVANT DOCUMENTS
# ==========================================

documents = retriever.invoke(
    question
)

print("Documents retrieved:", len(documents))


# ==========================================
# 9. AUGMENTATION
# ==========================================

context = "\n\n".join(
    document.page_content
    for document in documents
)

print("\nContext created!")


# ==========================================
# 10. PROMPT
# ==========================================

prompt = ChatPromptTemplate.from_template("""
You are a helpful YouTube video assistant.

Answer the question based ONLY on the context provided below.

If the answer is not present in the context,
say:

"I could not find the answer in the video."

Context:
{context}

Question:
{question}

Answer:
""")


# ==========================================
# 11. LOCAL LLM
# ==========================================

llm = ChatOllama(
    model="llama3.2"
)

print("Local LLM loaded!")


# ==========================================
# 12. BUILD CHAIN
# ==========================================

chain = prompt | llm

print("Chain created!")


# ==========================================
# 13. RUN CHAIN
# ==========================================

response = chain.invoke({
    "context": context,
    "question": question
})


# ==========================================
# 14. FINAL ANSWER
# ==========================================

print("\n===================================")
print("FINAL ANSWER")
print("===================================")

print(response.content)