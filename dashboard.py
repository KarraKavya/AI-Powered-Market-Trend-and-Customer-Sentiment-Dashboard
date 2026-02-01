import streamlit as st
import pandas as pd
import plotly.express as px

# LangChain + FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Gemini
from google import genai
from google.genai import types

# Utilities
import os
from dotenv import load_dotenv

# Scheduler (disabled safely)
import schedule
import threading
import time
# import news as news_service   # disabled (module not found)

# Groq
from groq import Groq
import notification

# ---------------------------------
# Load environment variables
# ---------------------------------
load_dotenv()

# ---------------------------------
# Streamlit Page Config
# ---------------------------------
st.set_page_config(
    page_title="AI Market Trend & Consumer Sentiment Forecaster",
    layout="wide"
)

st.title("AI-Powered Market Trend & Consumer Sentiment Dashboard")
st.markdown(
    "Consumer sentiment, topic trends, and social insights from **Reviews, Reddit, and News**"
)

# ---------------------------------
# Load Data
# ---------------------------------
@st.cache_data
def load_data():
    reviews = pd.read_csv(
        "final data/category_wise_lda_output_with_topic_labels.csv"
    )
    reddit = pd.read_excel(
        "final data/reddit_category_trend_data.xlsx"
    )
    news_df = pd.read_csv(
        "final data/news_data_with_sentiment.csv"
    )

    if "review_date" in reviews.columns:
        reviews["review_date"] = pd.to_datetime(reviews["review_date"], errors="coerce")

    if "created_date" in reddit.columns:
        reddit["created_date"] = pd.to_datetime(reddit["created_date"], errors="coerce")

    if "published_at" in news_df.columns:
        news_df["published_at"] = pd.to_datetime(news_df["published_at"], errors="coerce")

    return reviews, reddit, news_df

reviews_df, reddit_df, news_df = load_data()

# ---------------------------------
# Load FAISS Vector DB
# ---------------------------------
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.load_local(
        "consumer_sentiment_faiss1",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_db

vector_db = load_vector_db()

# ---------------------------------
# Load Gemini Client
# ---------------------------------
@st.cache_resource
def load_gemini_client():
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

client = load_gemini_client()

# ---------------------------------
# GROQ API CONFIG
# ---------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise Exception("GROQ API key not found in .env file")

groq_client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------
# Layout
# ---------------------------------
main_col, right_col = st.columns([3, 1])

# =====================================================================
# MAIN DASHBOARD
# =====================================================================
with main_col:

    # Sidebar Filters
    st.sidebar.header("Filters")

    source_filter = st.sidebar.multiselect(
        "Select Source",
        options=reviews_df["source"].unique(),
        default=reviews_df["source"].unique()
    )

    category_filter = st.sidebar.multiselect(
        "Select Category",
        options=reviews_df["category"].unique(),
        default=reviews_df["category"].unique()
    )

    filtered_reviews = reviews_df[
        (reviews_df["source"].isin(source_filter)) &
        (reviews_df["category"].isin(category_filter))
    ]

    # ---------------------------------
    # KPI Metrics
    # ---------------------------------
    st.subheader("Key Metrics")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Reviews", len(filtered_reviews))
    c2.metric("Positive %",
              round((filtered_reviews["sentiment_label"] == "Positive").mean() * 100, 1))
    c3.metric("Negative %",
              round((filtered_reviews["sentiment_label"] == "Negative").mean() * 100, 1))
    c4.metric("Neutral %",
              round((filtered_reviews["sentiment_label"] == "Neutral").mean() * 100, 1))

    # ---------------------------------
    # Sentiment Distribution
    # ---------------------------------
    col1, col2 = st.columns(2)

    with col1:
        sentiment_dist = filtered_reviews["sentiment_label"].value_counts().reset_index()
        sentiment_dist.columns = ["Sentiment", "Count"]
        fig = px.pie(sentiment_dist, names="Sentiment", values="Count", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cat_sent = (
            filtered_reviews.groupby(["category", "sentiment_label"])
            .size().reset_index(name="count")
        )
        fig = px.bar(cat_sent, x="category", y="count",
                     color="sentiment_label", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------
    # Sentiment Trend
    # ---------------------------------
    st.subheader("Sentiment Trend Over Time")

    trend = (
        filtered_reviews.groupby(
            [pd.Grouper(key="review_date", freq="W"), "sentiment_label"]
        ).size().reset_index(name="count")
    )

    fig = px.line(trend, x="review_date", y="count",
                  color="sentiment_label")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------
    # Category Trend
    # ---------------------------------
    st.subheader("Category Trend Over Time")

    cat_trend = (
        filtered_reviews.groupby(
            [pd.Grouper(key="review_date", freq="M"), "category"]
        ).size().reset_index(name="count")
    )

    fig = px.line(cat_trend, x="review_date", y="count",
                  color="category")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------
    # Topic Insights
    # ---------------------------------
    st.subheader("Topic Insights")

    topic_dist = filtered_reviews["topic_label"].value_counts().reset_index()
    topic_dist.columns = ["Topic", "Count"]
    fig = px.bar(topic_dist, x="Topic", y="Count")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------
    # Reddit Trends
    # ---------------------------------
    st.subheader("Reddit Category Popularity")

    reddit_trend = (
        reddit_df.groupby("category_label")
        .size().reset_index(name="mentions")
        .sort_values("mentions", ascending=False)
    )

    fig = px.bar(reddit_trend, x="category_label",
                 y="mentions")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------
    # News Sentiment
    # ---------------------------------
    st.subheader("News Sentiment Overview")

    news_sent = (
        news_df.groupby("sentiment_label")
        .size().reset_index(name="count")
    )

    fig = px.pie(news_sent, names="sentiment_label",
                 values="count")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------
    # Cross-Source Comparison
    # ---------------------------------
    st.subheader("Cross-Source Category Comparison")

    review_cat = reviews_df.groupby("category").size().reset_index(name="Review Mentions")
    reddit_cat = (
        reddit_df.groupby("category_label")
        .size().reset_index(name="Reddit Mentions")
        .rename(columns={"category_label": "category"})
    )
    news_cat = news_df.groupby("category").size().reset_index(name="News Mentions")

    compare = (
        review_cat.merge(reddit_cat, on="category", how="outer")
        .merge(news_cat, on="category", how="outer")
        .fillna(0)
    )

    fig = px.bar(compare, x="category",
                 y=["Review Mentions", "Reddit Mentions", "News Mentions"],
                 barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(compare)

# =====================================================================
# AI INSIGHT PANEL
# =====================================================================
with right_col:

    st.markdown("## 🤖 AI Insight Panel")

    llm_choice = st.radio(
        "Choose AI Model",
        ["Gemini", "Groq"],
        horizontal=True
    )

    user_query = st.text_area(
        "Your Question",
        height=140,
        placeholder="Ask about trends, sentiment, demand, risks..."
    )

    if st.button("Get Insight") and user_query:
        with st.spinner("Analyzing Market Intelligence..."):

            results = vector_db.similarity_search(user_query, k=8)
            context = "\n".join([r.page_content for r in results])

            prompt = f"""
You are a market intelligence analyst.
Use ONLY the provided context.

Context:
{context}

Question:
{user_query}

Answer:
"""

            if llm_choice == "Gemini":
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        thinking_config=types.ThinkingConfig(thinking_budget=0)
                    )
                )
                answer = response.text
            else:
                response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "Use only the given context."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                answer = response.choices[0].message.content

        st.success(f"Insight Generated using {llm_choice}")
        st.write(answer)
