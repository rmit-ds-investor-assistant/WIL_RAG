# pages/company_info.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Company Info")

st.title("🏢 Company Info")
st.caption("Quick reference of companies in the dataset")

# Optional: override/shorten descriptions here
COMPANY_DESCRIPTIONS = {
    # "RIO": "Diversified mining company focused on iron ore, aluminium, copper.",
    # "BHP": "Global resources company producing iron ore, copper, nickel.",
    # Add/modify as you wish...
}

@st.cache_data
def load_companies(path: str = "ragdata1.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path)
    # keep only useful columns if present
    cols = [
        "Company name",
        "Company Code",
        "Industry Group",
        "Company Description",
    ]
    existing = [c for c in cols if c in df.columns]
    df = df[existing].drop_duplicates(subset=[c for c in existing if c in ("Company Code","Company name")])
    # apply custom short descriptions (fallback to original)
    if "Company Code" in df.columns and "Company Description" in df.columns:
        df["Short Description"] = df.apply(
            lambda r: COMPANY_DESCRIPTIONS.get(r["Company Code"], r["Company Description"]),
            axis=1,
        )
    return df.sort_values(by=[c for c in ["Company name", "Company Code"] if c in df.columns])

df = load_companies()

# --- UI: search & list ---
q = st.text_input("Search by company, code, or industry")
if q:
    ql = q.lower()
    def hit(row):
        return any(ql in str(row.get(c, "")).lower() for c in df.columns)
    view = df[df.apply(hit, axis=1)]
else:
    view = df

st.write(f"Showing **{len(view)}** company record(s).")

# Render simple cards
for _, r in view.iterrows():
    with st.container(border=True):
        title = f"{r.get('Company name','?')}"
        code  = r.get("Company Code","?")
        industry = r.get("Industry Group","—")
        st.subheader(f"{title} ({code})")
        st.caption(f"Industry: {industry}")
        st.write(r.get("Short Description") or r.get("Company Description") or "No description provided.")

st.divider()
if st.button("⬅️ Back to Chat"):
    st.switch_page("app.py")
