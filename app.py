# app.py
import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  # builds/loads Chroma on import

st.set_page_config(page_title="Financial RAG Chatbot")

st.title("Financial RAG Chatbot (Ollama + Chroma)")
st.caption("""
Get to know the company before investing.

ABC is a smart, user-friendly chat that helps investors to explore and understand 
           companies before investing. Make informed decisions by reviewing the data. 

Ask your question on company here. 

Disclaimer: ABC provides company data and insights for informational purposes only. 
           We do not offer financial, investment, or legal advice. 
           Users should conduct their own research and seek professional guidance before making investment decisions. 
           ABC is not responsible for any financial losses or actions taken based on the information provided.
""")

# Model + prompt (cache the chain so it’s created once)
@st.cache_resource
def get_chain(model_name: str = "llama3.2"):
    model = OllamaLLM(model=model_name)
    template = """
You are an expert financial analyst.
Answer the question based on the company financial records below.

Relevant records:
{reviews}

Question: {question}

If calculations are needed, explain step by step before giving the final answer.
"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model

def format_docs(docs):
    # Pretty print retrieved docs for the prompt
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        header = f"[Doc {i}] Company Code: {meta.get('company_code','?')} | Industry: {meta.get('industry','?')}"
        parts.append(header + "\n" + d.page_content)
    return "\n\n---\n\n".join(parts)

# Sidebar controls
with st.sidebar:

    st.subheader("Navigation")
    if st.button("Chat"):
        st.switch_page("app.py")

    if st.button("Company Info"):
        st.switch_page("pages/company_info.py")


    st.subheader("Settings")
    model_name = st.text_input("Ollama model", value="llama3.2")
    # top_k = st.slider("Retriever k", min_value=1, max_value=10, value=5)
    st.markdown(
        "Make sure you’ve pulled the models locally:\n\n"
        "`ollama pull llama3.2`\n\n`ollama pull mxbai-embed-large`"
    )

    

# Allow changing k at runtime
# retriever.search_kwargs["k"] = top_k

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask about a company’s performance, revenue, profit, etc.")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            docs = retriever.invoke(question)
            reviews = format_docs(docs)
            chain = get_chain(model_name)
            answer = chain.invoke({"reviews": reviews, "question": question})

            st.markdown(answer)

            with st.expander("Show retrieved records"):
                for i, d in enumerate(docs, 1):
                    meta = d.metadata or {}
                    st.markdown(
                        f"**Doc {i}** — "
                        f"Company: {meta.get('company_name','?')} "
                        f"(Code: {meta.get('company_code','?')}) • "
                        f"Industry: {meta.get('industry','?')}"
                    )
                    st.text(d.page_content[:1500])  # preview

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Utility buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🔄 Reset chat"):
        st.session_state.messages = []
        st.rerun()
with col2:
    st.write("")  # spacer
