from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever, df
import matplotlib.pyplot as plt

model = OllamaLLM(model="llama3.2")

template = """
You are an expert financial analyst. 
Answer the question based on the company financial reports below.

Here are some relevant company records:{reviews}

Here is the question to answer: {question}

If the questions is not relevant to financial reports, please reply accordingly.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


# ---------------- Chart Function ----------------
def plot_revenue(df, question: str, company: str = None):
    if company:
        data = df[df["Company name"].str.contains(company, case=False)]
        if data.empty:
            print(f"No data found for {company}")
            return
    else:
        data = df

    # detect which years to show
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
    plt.show()


while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    
     # Chart detection
    if any(k in question.lower() for k in ["chart", "plot", "graph", "visualize"]):
        print("Chart branch triggered!")
        company = None
        for name in df["Company name"].unique():
            if name.lower() in question.lower():
                company = name
                break
        plot_revenue(df, company)
        continue



    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)