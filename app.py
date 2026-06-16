import streamlit as st
import pdfplumber
import requests
import datetime
import random

# -------------------- BASIC SETUP --------------------
st.set_page_config(page_title="Services 4 Digital AI Firm", layout="centered")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
PDF_PATH = "data/Company_Manual.pdf" 
MODEL_NAME = "llama-3.1-8b-instant"

# -------------------- LOAD + CHUNK PDF --------------------
@st.cache_data
def load_chunks(max_chars: int = 600):
    text = ""

    try:
        with pdfplumber.open(PDF_PATH) as pdf:
            for page in pdf.pages:
                tx = page.extract_text()
                if tx:
                    text += tx + "\n"
    except:
        return []

    raw_parts = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    buf = ""

    for part in raw_parts:
        if len(buf) + len(part) <= max_chars:
            buf += " " + part
        else:
            chunks.append(buf.strip())
            buf = part

    if buf:
        chunks.append(buf.strip())

    return chunks


pdf_chunks = load_chunks()

# -------------------- SIMPLE RETRIEVAL --------------------
def retrieve_context(query: str, top_k: int = 3):
    q_words = set(query.lower().split())
    scored = []

    for ch in pdf_chunks:
        ch_words = set(ch.lower().split())
        score = len(q_words & ch_words)

        if score > 0:
            scored.append((score, ch))

    if not scored:
        return ""

    scored.sort(reverse=True, key=lambda x: x[0])
    return "\n\n".join(c for _, c in scored[:top_k])


# -------------------- GROQ API CALL --------------------
def llama_chat(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.4,
    }

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=60
        )

        result = response.json()

        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"⚠️ Groq API Error:\n{str(e)}"


# -------------------- RAG + UPDATED INFO --------------------
def get_answer(question: str, history):
    context = retrieve_context(question)
    today = datetime.datetime.now().strftime("%d %B %Y (%Y)")
    pdf_strength = len(context.strip())

    if pdf_strength < 50:

        system_prompt = f"""
You are S4D Chatbot.

Rules:
- Give clear and direct answers.
- Use your latest general knowledge.
- Today's date is {today}.
- Never say you are searching or researching.
- Be helpful, friendly and confident.
"""

    else:

        system_prompt = f"""
You are Joti Chatbot also Mmanaged by Joti.

Use the PDF context as your primary source.

PDF Context:
---------------------
{context}
---------------------

Rules:
- Answer confidently.
- Use PDF information first.
- If needed, combine with updated knowledge.
- Never say you are searching or researching.
- Today's date is {today}.
"""

    messages = [{"role": "system", "content": system_prompt}]

    for msg in history[-6:]:
        messages.append(msg)

    messages.append({"role": "user", "content": question})

    return llama_chat(messages)


# -------------------- STREAMLIT UI --------------------
st.title("🤖 S4D Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content":
            "Assalam o Alaikum! 👋 Main S4D ka Chatbot hoon.\n\n"
            "Jo bhi poochna hai bindaas poochho, main aapki madad ke liye hamesha tayyar hoon."
        }
    ]

# Display old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
user_input = st.chat_input("Apna sawal likho...")

if user_input:

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Soch raha hoon..."):
            answer = get_answer(
                user_input,
                st.session_state.messages
            )

        st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
