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
        Information: {row['Additional Information']}
        Website: {row['Company Website']}
        ASX: {row['Company page in ASX']}
        Revenue (H1 2025) in AUD: {row['Half year ending June 2025 Revenue']}
        Revenue (H1 2024) in AUD: {row['Half year ending June 2024 Revenue']}
        Revenue Change: {row['Revenue Half year Percentage Change']}
        Profit After Tax (H1 2025) in AUD: {row['Half year ending June 2025 Profit after tax attributable to shareholders (net earnings) in Mn']}
        Profit After Tax (H1 2024) in AUD: {row['Half year ending June 2024 Profit after tax attributable to shareholders (net earnings) in Mn']}
        Profit Change H1: {row['Profit after tax attributable to shareholders (net earnings) Percentage Change']}
        Revenue (Full Year 2025): {row['Revenue for full year ending Jun 25 in mn in AUD']}       
        Revenue (Full Year 2024): {row['Revenue for full year ending Jun 24 in mn in AUD']}
        Revenue Change Full Year: {row['Revenue for full year percentage change']}
        Profit After Tax (Full Year 2025) in AUD: {row['Full year ending June 2025 Profit after tax attributable to shareholders (net earnings) in Mn']}
        Profit After Tax (Full Year 2025) in AUD: {row['Full year ending June 2024 Profit after tax attributable to shareholders (net earnings) in Mn']}
        Profit Change Full Year: {row['Full year ending June 2024 Profit after tax attributable to shareholders percentage change']}
        Equity: {row['Equity for shareholder in Mn AUD']}
        Shares: {row['Number of Shares in Millions']}
        Market price: {row['Market Price in AUD on 15th Sep  2025']}
        EPS (H1 2025): {row['Earnings per Share (AUD) for half year ending June 2025 (Net Profit attributable to share holder) divided by Number of shares']}
        EPS (H1 2024): {row['Earnings per Share (AUD) for half year ending June 2024 (Net Profit attributable to share holder) divided by Number of shares. EPS measures how much profit a company generates for each outstanding share of its stock. Interpretation: A higher EPS generally means the company is more profitable on a per-share basis, which is positive for shareholders. ']}
        Price-to-Earnings (P/E, H1 2024): {row['Price to Earnings ratio (x) for half year ending June 2025 : Market price divided by Earnings per share. The PE ratio shows how much investors are willing to pay for each dollar of the companys earnings. It compares the companys share price with its earnings per share. Interpretation: A high P/E ratio suggests the market expects strong future growth (but it could also mean the stock is overpriced). A low PE ratio may indicate undervaluation or slower growth expectations. ']}
        Book Value per Share (BVPS): {row['Book value per share (AUD): Equity for shareholder/Number of shares. Book Value per Share (BVPS) is a financial ratio that represents the net asset value of a company available to each outstanding share of common stock. BVPS shows the accounting value of each share if the company were liquidated today, based on its historical cost of assets and liabilities. Comparison with Market Price: Market Price per Share > BVPS: Investors believe the company has strong future earnings power, intangible assets (brand, patents, goodwill), or growth potential not fully reflected on the balance sheet. Market Price per Share < BVPS: Could indicate the stock is undervalued, or it may signal concerns about profitability, asset quality, or growth. Indicator of Financial Strength: A higher BVPS over time suggests the company is consistently building value for shareholders (through retained earnings, reinvestments). A declining BVPS could mean losses, high dividend payouts, or write-downs of assets.']}
        Price-to-Book Ratio (P/B): {row['Price to Book value (x): Market price/Book value per share: It shows how much investors are willing to pay for each $1 of book value (net assets) of the company.']}
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

# Export retriever + dataframe
__all__ = ["retriever", "df"]