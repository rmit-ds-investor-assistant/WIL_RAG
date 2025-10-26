# FinGenie

# File Layout

WIL_RAG/
│
├── .venv/ # Virtual environment (not committed to Git)
├── .streamlit/
│ └── config.toml # Streamlit configuration file
│
├── pages/
│ └── company_info.py # Streamlit page to browse company info
│
├── app.py # Main Streamlit chatbot interface
├── main.py # CLI testing version (optional)
├── vector.py # Script to build/load Chroma DB
├── ragdata1.xlsx # Company financial dataset
├── fingenie_logo.png # App logo for sidebar
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # (Optional) ignore venv, cache, DB, etc.