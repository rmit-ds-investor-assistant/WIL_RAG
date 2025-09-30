# app.py
import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever, df  # builds/loads Chroma on import
import matplotlib.pyplot as plt

st.set_page_config(page_title="Financial Chatbot")

st.title("FinGenie")
st.caption("""
Get to know the company before investing.

FinGenie is a smart, user-friendly chat that helps investors to explore and understand 
           companies before investing. Make informed decisions by reviewing the data. 

Ask your question on company here. 

Disclaimer: FinGenie provides company data and insights for informational purposes only. 
           We do not offer financial, investment, or legal advice. 
           Users should conduct their own research and seek professional guidance before making investment decisions. 
           FinGenie is not responsible for any financial losses or actions taken based on the information provided.
""")

# Model + prompt (cache the chain so it’s created once)
@st.cache_resource
def get_chain(model_name: str = "llama3.2"):
    model = OllamaLLM(model=model_name)
    template = """
You are an expert financial analyst. 
Answer the question based on the company financial reports below.

Here are some relevant company records:{reviews}

Here is the question to answer: {question}

If the questions is not relevant to financial reports, please reply accordingly.
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


# ---------------- Chart Function ----------------
def plot_revenue(df, question: str, company: str = None):
    if company:
        data = df[df["Company name"].str.contains(company, case=False)]
        if data.empty:
            st.warning(f"No data found for {company}")
            return None
    else:
        data = df

    show_2024 = "2024" in question
    show_2025 = "2025" in question
    if not show_2024 and not show_2025:
        show_2024, show_2025 = True, True

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(data))
    width = 0.35

    if show_2024:
        ax.bar(
            [i - width/2 if show_2025 else i for i in x],
            data["Half year ending June 2024 Revenue"],
            width if show_2025 else 0.6,
            label="Revenue H1 2024"
        )

    if show_2025:
        ax.bar(
            [i + width/2 if show_2024 else i for i in x],
            data["Half year ending June 2025 Revenue"],
            width if show_2024 else 0.6,
            label="Revenue H1 2025"
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(data["Company name"], rotation=45, ha="right")
    ax.set_ylabel("Revenue (mn AUD)")
    ax.set_title("Half-Year Revenue")
    ax.legend()
    plt.tight_layout()
    return fig



# Sidebar controls
with st.sidebar:
    st.image("fingenie_logo.png", width='stretch')

    st.subheader("Navigation")
    if st.button("Chat"):
        st.switch_page("app.py")

    if st.button("Company Info"):
        st.switch_page("pages/company_info.py")


    st.subheader("Settings")
    model_name = st.text_input("Ollama model", value="llama3.2")
    # top_k = st.slider("Retriever k", min_value=1, max_value=10, value=5)
    # st.markdown(
    #     "Make sure you’ve pulled the models locally:\n\n"
    #     "`ollama pull llama3.2`\n\n`ollama pull mxbai-embed-large`"
    # )

    

# Allow changing k at runtime
# retriever.search_kwargs["k"] = top_k

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "chart" in m:  # show stored chart
            st.pyplot(m["chart"])

# ---------------- Handle New User Input ----------------
question = st.chat_input("Ask about a company’s performance, revenue, profit, etc.")
if question:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Assistant response
    if any(k in question.lower() for k in ["chart", "plot", "graph", "visualize"]):
        company = None
        for name in df["Company name"].dropna().astype(str).unique():
            if name.lower() in question.lower():
                company = name
                break
        fig = plot_revenue(df, question, company)
        if fig:
            msg = {"role": "assistant", "content": "Here is the bar chart you requested 📊", "chart": fig}
            st.session_state.messages.append(msg)
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                st.pyplot(fig)
        else:
            msg = {"role": "assistant", "content": "Sorry, I couldn’t generate a chart for that query."}
            st.session_state.messages.append(msg)
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = retriever.invoke(question)
                reviews = format_docs(docs)
                chain = get_chain("llama3.2")
                answer = chain.invoke({"reviews": reviews, "question": question})

                st.markdown(answer)

                # 🆕 Show retrieved records only for normal questions
                with st.expander("📂 Show retrieved records"):
                    for i, d in enumerate(docs, 1):
                        meta = d.metadata or {}
                        st.markdown(
                            f"**Doc {i}** — "
                            f"Company Code: {meta.get('company_code','?')} | "
                            f"Industry: {meta.get('industry','?')}"
                        )
                        st.text(d.page_content[:1200])  # preview (cut off long text)

                st.session_state.messages.append({"role": "assistant", "content": answer})
   

# ---------------- Reset Chat ----------------
# if st.button("🔄 Reset chat"):
#     st.session_state.messages = []
#     st.rerun()
