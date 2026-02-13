import streamlit as st
import sys
import os
import html

# Add parent directory to path to import model_service
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_service import get_model_service

st.set_page_config(page_title="SentiMind AI", page_icon="🧠", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #f8fafc;
    }

    .logo-text {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4f46e5;
    }

    .sentiment-badge {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.5rem;
        font-weight: 700;
    }

    .sentiment-positive { color: #10b981; }
    .sentiment-negative { color: #ef4444; }
    .sentiment-neutral { color: #6b7280; }

    .confidence-text {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }

    .confidence-value {
        font-size: 1.75rem;
        font-weight: 700;
    }

    .history-item {
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid #eef2f7;
        margin-bottom: 0.75rem;
        background-color: #f1f5f9;
    }

    .score-circle {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        border: 12px solid #e2e8f0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        position: relative;
        margin: 0 auto;
    }

    .score-value {
        font-size: 2.25rem;
        font-weight: 700;
    }

    /* Custom button styling */
    div.stButton > button[data-testid="stBaseButton-primary"] {
        background-color: #ccd3e2 !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 2rem !important;
    }

    div.stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: #b5bdd1 !important;
    }

    /* Subtle Clear button */
    button[key="clear_all_btn"] {
        background: none !important;
        border: none !important;
        color: #94a3b8 !important;
        font-size: 0.75rem !important;
        text-decoration: none !important;
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: none !important;
        height: auto !important;
        line-height: normal !important;
        min-height: unset !important;
    }

    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_analyses' not in st.session_state:
    st.session_state.total_analyses = 0

# Top Header
head_col1, head_col2 = st.columns([5, 1])
with head_col1:
    st.markdown('<div style="display: flex; align-items: center; gap: 0.5rem;"><span style="font-size: 2rem;">🧠</span><span class="logo-text">SentiMind AI</span></div>', unsafe_allow_html=True)
with head_col2:
    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_analyses = 0
        st.rerun()

# Main Layout
main_col, history_col = st.columns([2.2, 1])

service = get_model_service()

with main_col:
    # Analyze New Sentiment Card
    with st.container(border=True):
        st.markdown("""
        <h3 style="margin-top: 0; display: flex; align-items: center; gap: 0.5rem; font-size: 1.25rem;">
            <span style="color: #4f46e5; font-size: 1.5rem;">📈</span> Analyze New Sentiment
        </h3>
        <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 1rem;">Enter text below to uncover hidden emotions and sentiment patterns.</p>
        """, unsafe_allow_html=True)

        user_input = st.text_area("Input", placeholder="What's on your mind? Paste a review, a tweet, or a paragraph...", label_visibility="collapsed", height=120)

        btn_col1, btn_col2 = st.columns([4.5, 1])
        with btn_col2:
            analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

    if analyze_clicked:
        if not user_input.strip():
            st.warning("Please enter some text.")
        elif service.model is None:
            st.error("Model is not available. Please ensure it is trained.")
        else:
            with st.spinner("Analyzing..."):
                result = service.predict(user_input)

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state.total_analyses += 1
                    # Escape input for history and display
                    safe_input = html.escape(user_input)
                    st.session_state.history.insert(0, {
                        "text": safe_input[:100] + "..." if len(safe_input) > 100 else safe_input,
                        "sentiment": result["overall_sentiment"],
                        "confidence": result["top_prediction"]["confidence"]
                    })

                    st.markdown(f"### Analysis Results")

                    res_col1, res_col2 = st.columns([1.4, 1])

                    with res_col1:
                        sentiment_class = f"sentiment-{result['overall_sentiment'].lower()}"
                        icon = "😊" if result['overall_sentiment'] == "Positive" else "😔" if result['overall_sentiment'] == "Negative" else "😐"
                        accent_color = "#10b981" if result['overall_sentiment'] == "Positive" else "#ef4444" if result['overall_sentiment'] == "Negative" else "#6b7280"

                        st.markdown(f"""
                        <div style="background-color: white; padding: 1.5rem; border-radius: 1rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border: 1px solid #e2e8f0; border-left: 5px solid {accent_color};">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                <div>
                                    <div class="confidence-text">OVERALL SENTIMENT</div>
                                    <div class="sentiment-badge {sentiment_class}">{icon} {result['overall_sentiment']}</div>
                                </div>
                                <div style="text-align: right;">
                                    <div class="confidence-text">CONFIDENCE</div>
                                    <div class="confidence-value" style="color: #4f46e5;">{result['top_prediction']['confidence']:.0%}</div>
                                </div>
                            </div>
                            <div style="background-color: #f8fafc; padding: 1.25rem; border-radius: 0.75rem; margin: 1.5rem 0; font-style: italic; color: #334155; border: 1px solid #f1f5f9; line-height: 1.6;">
                                "{safe_input}"
                            </div>
                            <div style="font-weight: 700; margin-bottom: 0.5rem; font-size: 0.9rem;">Key Logic</div>
                            <div style="color: #64748b; font-size: 0.85rem; line-height: 1.6;">
                                {html.escape(result['key_logic'])}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with res_col2:
                        score = result['sentiment_score']
                        score_color = "#10b981" if score > 0.1 else "#ef4444" if score < -0.1 else "#6b7280"

                        st.markdown(f"""
                        <div style="background-color: white; padding: 1.5rem; border-radius: 1rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border: 1px solid #e2e8f0; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 300px;">
                            <div class="confidence-text" style="margin-bottom: 1.5rem; text-align: center;">Sentiment Score</div>
                            <div class="score-circle" style="border-top-color: {score_color}; border-right-color: {score_color}; border-bottom-color: {score_color};">
                                <div class="score-value" style="color: {score_color};">{"+" if score > 0 else ""}{score:.2f}</div>
                                <div class="score-label" style="margin-top: 0.25rem;">-1 TO 1 RANGE</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

with history_col:
    with st.container(border=True):
        hist_header_col1, hist_header_col2 = st.columns([3, 1])
        with hist_header_col1:
            st.markdown("""
                <h3 style="margin: 0; font-size: 1.15rem; display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #4f46e5; font-size: 1.4rem;">🕒</span> History
                </h3>
            """, unsafe_allow_html=True)
        with hist_header_col2:
            if st.button("Clear all", key="clear_all_btn"):
                st.session_state.history = []
                st.rerun()

        st.markdown('<div style="margin-top: 1rem; max-height: 500px; overflow-y: auto;">', unsafe_allow_html=True)
        for item in st.session_state.history:
            icon = "😊" if item['sentiment'] == "Positive" else "😔" if item['sentiment'] == "Negative" else "😐"
            color = "#10b981" if item['sentiment'] == "Positive" else "#ef4444" if item['sentiment'] == "Negative" else "#6b7280"
            st.markdown(f"""
            <div class="history-item">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span style="font-size: 1rem;">{icon}</span>
                    <span style="font-weight: 700; font-size: 0.7rem; color: {color}; text-transform: uppercase; letter-spacing: 0.02em;">{item['sentiment']}</span>
                </div>
                <div style="font-size: 0.8rem; color: #475569; font-style: italic; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">
                    "{item['text']}"
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #f1f5f9; display: flex; justify-content: space-between; align-items: center; color: #94a3b8; font-size: 0.75rem; font-weight: 700;">
                <span>TOTAL ANALYSES</span>
                <span>{st.session_state.total_analyses}</span>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with HuggingFace Transformers and Streamlit.")
