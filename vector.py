# from langchain_ollama import OllamaEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.documents import Document
# import os
# import pandas as pd

# df = pd.read_csv("realistic_restaurant_reviews.csv")
# embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# db_location = "./chrome_langchain_db"
# add_documents = not os.path.exists(db_location)

# if add_documents:
#     documents = []
#     ids = []
    
#     for i, row in df.iterrows():
#         document = Document(
#             page_content=row["Title"] + " " + row["Review"],
#             metadata={"rating": row["Rating"], "date": row["Date"]},
#             id=str(i)
#         )
#         ids.append(str(i))
#         documents.append(document)
        
# vector_store = Chroma(
#     collection_name="restaurant_reviews",
#     persist_directory=db_location,
#     embedding_function=embeddings
# )

# if add_documents:
#     vector_store.add_documents(documents=documents, ids=ids)
    
# retriever = vector_store.as_retriever(
#     search_kwargs={"k": 5}
# )

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Load dataset
df = pd.read_excel("ragdata1.xlsx")

# Embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Persistent vector DB
db_location = "./chroma_company_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        # Convert row into a document
        text_content = f"""
        Company: {row['Company name']} (Code: {row['Company Code']})
        Industry: {row['Industry Group']}
        Description: {row['Company Description']}
        Revenue (H1 2025): {row['Half year ending June 2025 Revenue']}
        Revenue (H1 2024): {row['Half year ending June 2024 Revenue']}
        Revenue Change: {row['Revenue Percentage Change']}
        Profit After Tax (H1 2025): {row['Half year ending June 2025 Profit after tax attributable to shareholders (net earnings)']}
        Profit After Tax (H1 2024): {row['Half year ending June 2024 Profit after tax attributable to shareholders (net earnings)']}
        Profit Change: {row['Profit after tax attributable to shareholders (net earnings) Percentage Change']}
        """
        document = Document(
            page_content=text_content.strip(),
            metadata={"company_code": row["Company Code"], "industry": row["Industry Group"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

# Create / load Chroma DB
vector_store = Chroma(
    collection_name="company_financials",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Retriever with top-k docs
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
