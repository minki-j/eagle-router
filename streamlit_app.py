"""
Streamlit app for Eagle Router model selection and feedback
"""

import streamlit as st
import os
import numpy as np
from openai import OpenAI
from eagle_router import EagleRouter
import logging
from dotenv import load_dotenv
from typing import Optional, cast

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(page_title="Eagle Router Demo", page_icon="🦅", layout="wide")

AVAILABLE_MODELS = ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06"]

# Custom styles
st.markdown(
    """
    <style>
    /* Ensure consistent button heights */
    div.stButton > button {
        height: 2.5rem;
    }
    /* Style primary button */
    div.stButton > button[kind="primary"] {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Please set OPENAI_API_KEY in your .env file")
        st.stop()
    return OpenAI(api_key=api_key)


# Initialize router with configurable parameters
@st.cache_resource
def get_router(p_value, n_value, k_value):
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        st.error("Please set MONGODB_URI in your .env file")
        st.stop()

    return EagleRouter(
        mongodb_uri=mongodb_uri,
        P=p_value,
        N=n_value,
        K=k_value,
    )


# Get embedding for a text
def get_embedding(text: str, client: OpenAI) -> np.ndarray:
    """Get embedding for a text using OpenAI's API"""
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(response.data[0].embedding)


# Get model response
def get_model_response(prompt: str, model: str, client: OpenAI) -> str:
    """Get response from a specific model"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000,
        )
        content = response.choices[0].message.content
        return content or ""
    except Exception as e:
        return f"Error getting response from {model}: {str(e)}"


def get_short_model_name(model: Optional[str]) -> str:
    """Return a short, user-friendly model name."""
    if not model:
        return "Unknown"
    return model.split("-20")[0]


# Main app
def main():
    st.title("🦅 Eagle Router")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Define available parameter values
        P_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        N_VALUES = [5, 10, 20, 30, 50]
        K_VALUES = [1, 2, 4, 8, 16, 32, 64]

        p_value = st.selectbox(
            "P (Global vs Local weight)",
            options=P_VALUES,
            index=P_VALUES.index(0.3),  # Default to 0.3
            help="Weight for global scores (0=pure local, 1=pure global)",
        )

        n_value = st.selectbox(
            "N (Nearest neighbors)",
            options=N_VALUES,
            index=N_VALUES.index(10),  # Default to 10
            help="Number of nearest neighbors for local scoring",
        )

        k_value = st.selectbox(
            "K (ELO sensitivity)",
            options=K_VALUES,
            index=K_VALUES.index(8),  # Default to 8
            help="ELO rating adjustment factor",
        )

        st.divider()

    # Initialize clients and router
    client = get_openai_client()
    router = get_router(p_value, n_value, k_value)

    # Sidebar: Current rankings (smaller font)
    with st.sidebar:
        if router.elo_scores:
            st.subheader("📊 Model Rankings")
            for idx, (model, score) in enumerate(
                sorted(router.elo_scores.items(), key=lambda x: x[1], reverse=True)
            ):
                st.caption(f"{idx + 1}. {model.split('-20')[0]} (ELO: {score:.1f})")

    # Initialize session state
    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = ""
    if "current_embedding" not in st.session_state:
        st.session_state.current_embedding = None
    if "best_model" not in st.session_state:
        st.session_state.best_model = None
    if "other_model" not in st.session_state:
        st.session_state.other_model = None
    if "response_best" not in st.session_state:
        st.session_state.response_best = None
    if "response_other" not in st.session_state:
        st.session_state.response_other = None
    if "show_comparison" not in st.session_state:
        st.session_state.show_comparison = False

    prompt = st.text_area(
        "Type your prompt here:",
        height=100,
        placeholder="Ask anything... The router will select the best model for your query.",
    )

    # Option 1: Buttons in the same column, side by side
    col1, col2 = st.columns([8, 4])
    with col2:
        button_col1, button_col2 = st.columns(2, gap="small")
        with button_col1:
            # Disable compare button if no response is available yet or already comparing
            compare_button = st.button(
                "Compare",
                use_container_width=True,
                help="Get a response first to enable comparison"
                if st.session_state.response_best is None
                else "Already comparing models"
                if st.session_state.show_comparison
                else "Compare with another model",
            )
        with button_col2:
            submit_button = st.button(
                "Get Response", type="primary", use_container_width=True
            )

    # Process prompt submission
    if submit_button and prompt:
        with st.spinner("Processing your request..."):
            # Reset state for new prompt
            st.session_state.current_prompt = prompt
            st.session_state.response_other = None
            st.session_state.show_comparison = False

            # Get embedding
            st.session_state.current_embedding = get_embedding(prompt, client)

            # Get router recommendation
            models = AVAILABLE_MODELS
            scores = router.route(st.session_state.current_embedding, models)

            # Determine best and other model
            sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            st.session_state.best_model = sorted_models[0][0]
            st.session_state.other_model = sorted_models[1][0]

            # Get response from best model
            st.session_state.response_best = get_model_response(
                prompt, st.session_state.best_model, client
            )

    # Handle comparison request
    if compare_button:
        with st.spinner(
            f"Getting response from {get_short_model_name(st.session_state.other_model)}..."
        ):
            st.session_state.response_other = get_model_response(
                st.session_state.current_prompt,
                cast(str, st.session_state.other_model),
                client,
            )
            st.session_state.show_comparison = True
            st.rerun()

    # Display single response
    if st.session_state.response_best and not st.session_state.show_comparison:
        # Show model selection info
        st.text(get_short_model_name(st.session_state.best_model))

        # Show response
        st.markdown(st.session_state.response_best)

    # Display comparison and feedback
    if st.session_state.show_comparison and st.session_state.response_other:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{get_short_model_name(st.session_state.best_model)}**")
            container1 = st.container(border=True)
            with container1:
                st.markdown(st.session_state.response_best)
            if st.button("👍 This is better", use_container_width=True, key="vote_a"):
                # Model A (best_model) wins
                best_model = st.session_state.best_model or ""
                other_model = st.session_state.other_model or ""
                score = 0 if best_model < other_model else 1
                router.add_new_match(
                    prompt=st.session_state.current_prompt,
                    embedding=cast(np.ndarray, st.session_state.current_embedding),
                    model_a=min(best_model, other_model),
                    model_b=max(best_model, other_model),
                    score=score,
                )
                st.success("✅ Feedback recorded! Router updated.")
                st.balloons()
                # Reset comparison state
                st.session_state.show_comparison = False
                st.session_state.response_other = None
                st.rerun()

        with col2:
            st.markdown(f"**{get_short_model_name(st.session_state.other_model)}**")
            container2 = st.container(border=True)
            with container2:
                st.markdown(st.session_state.response_other)
            if st.button("👍 This is better", use_container_width=True, key="vote_b"):
                # Model B (other_model) wins
                best_model = st.session_state.best_model or ""
                other_model = st.session_state.other_model or ""
                score = 1 if best_model < other_model else 0
                router.add_new_match(
                    prompt=st.session_state.current_prompt,
                    embedding=cast(np.ndarray, st.session_state.current_embedding),
                    model_a=min(best_model, other_model),
                    model_b=max(best_model, other_model),
                    score=score,
                )
                st.success("✅ Feedback recorded! Router updated.")
                st.balloons()
                # Reset comparison state
                st.session_state.show_comparison = False
                st.session_state.response_other = None
                st.rerun()

        # Draw option
        col_center = st.columns([3, 2, 3])[1]
        with col_center:
            if st.button("🤝 Both are equally good", use_container_width=True):
                best_model = st.session_state.best_model or ""
                other_model = st.session_state.other_model or ""
                router.add_new_match(
                    prompt=st.session_state.current_prompt,
                    embedding=cast(np.ndarray, st.session_state.current_embedding),
                    model_a=min(best_model, other_model),
                    model_b=max(best_model, other_model),
                    score=2,  # Draw
                )
                st.success("✅ Feedback recorded as a draw! Router updated.")
                # Reset comparison state
                st.session_state.show_comparison = False
                st.session_state.response_other = None
                st.rerun()

    # Rankings moved to sidebar above


if __name__ == "__main__":
    main()
