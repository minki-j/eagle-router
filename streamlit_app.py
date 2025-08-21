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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(page_title="Eagle Router Demo", page_icon="🦅", layout="wide")


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
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting response from {model}: {str(e)}"


# Main app
def main():
    st.title("🦅 Eagle Router Demo")
    st.markdown("### Intelligent Model Routing with User Feedback")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Router Parameters")
        p_value = st.slider(
            "P (Global vs Local weight)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Weight for global scores (0=pure local, 1=pure global)",
        )

        n_value = st.slider(
            "N (Nearest neighbors)",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Number of nearest neighbors for local scoring",
        )

        k_value = st.slider(
            "K (ELO sensitivity)",
            min_value=1,
            max_value=32,
            value=8,
            step=1,
            help="ELO rating adjustment factor",
        )

        st.divider()

        st.subheader("Available Models")
        models = ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06"]
        for model in models:
            st.info(f"• {model.split('-20')[0]}")

        if st.button("🔄 Reset Router Cache"):
            st.cache_resource.clear()
            st.success("Router cache cleared!")
            st.rerun()

    # Initialize clients and router
    client = get_openai_client()
    router = get_router(p_value, n_value, k_value)

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

    # Main input area
    st.header("💬 Enter Your Prompt")

    prompt = st.text_area(
        "Type your prompt here:",
        height=100,
        placeholder="Ask anything... The router will select the best model for your query.",
    )

    col1, col2, col3 = st.columns([2, 2, 8])
    with col1:
        submit_button = st.button(
            "🚀 Get Response", type="primary", use_container_width=True
        )
    with col2:
        if st.session_state.response_best and not st.session_state.show_comparison:
            compare_button = st.button("🔄 Compare Models", use_container_width=True)
        else:
            compare_button = False

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
            models = ["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06"]
            scores = router.route(st.session_state.current_embedding, models)

            # Determine best and other model
            sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            st.session_state.best_model = sorted_models[0][0]
            st.session_state.other_model = sorted_models[1][0]

            # Get response from best model
            st.session_state.response_best = get_model_response(
                prompt, st.session_state.best_model, client
            )

    # Display single response
    if st.session_state.response_best and not st.session_state.show_comparison:
        st.divider()
        st.subheader("📝 Response")

        # Show model selection info
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Selected Model", st.session_state.best_model.split("-20")[0])
        with col2:
            if st.session_state.best_model and st.session_state.other_model:
                scores = router.route(
                    st.session_state.current_embedding,
                    [st.session_state.best_model, st.session_state.other_model],
                )
                score_diff = (
                    scores[st.session_state.best_model]
                    - scores[st.session_state.other_model]
                )
                st.metric("Confidence", f"{score_diff:.2f} points ahead")

        # Show response
        st.markdown(st.session_state.response_best)

    # Handle comparison request
    if compare_button:
        with st.spinner(
            f"Getting response from {st.session_state.other_model.split('-20')[0]}..."
        ):
            st.session_state.response_other = get_model_response(
                st.session_state.current_prompt, st.session_state.other_model, client
            )
            st.session_state.show_comparison = True
            st.rerun()

    # Display comparison and feedback
    if st.session_state.show_comparison and st.session_state.response_other:
        st.divider()
        st.subheader("🤔 Model Comparison - Which response is better?")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Model A: {st.session_state.best_model.split('-20')[0]}**")
            container1 = st.container(border=True)
            with container1:
                st.markdown(st.session_state.response_best)
            if st.button(
                "👍 Model A is better", use_container_width=True, key="vote_a"
            ):
                # Model A (best_model) wins
                score = (
                    0
                    if st.session_state.best_model < st.session_state.other_model
                    else 1
                )
                router.add_new_match(
                    prompt=st.session_state.current_prompt,
                    embedding=st.session_state.current_embedding,
                    model_a=min(
                        st.session_state.best_model, st.session_state.other_model
                    ),
                    model_b=max(
                        st.session_state.best_model, st.session_state.other_model
                    ),
                    score=score,
                )
                st.success("✅ Feedback recorded! Router updated.")
                st.balloons()
                # Reset comparison state
                st.session_state.show_comparison = False
                st.session_state.response_other = None
                st.rerun()

        with col2:
            st.markdown(f"**Model B: {st.session_state.other_model.split('-20')[0]}**")
            container2 = st.container(border=True)
            with container2:
                st.markdown(st.session_state.response_other)
            if st.button(
                "👍 Model B is better", use_container_width=True, key="vote_b"
            ):
                # Model B (other_model) wins
                score = (
                    1
                    if st.session_state.best_model < st.session_state.other_model
                    else 0
                )
                router.add_new_match(
                    prompt=st.session_state.current_prompt,
                    embedding=st.session_state.current_embedding,
                    model_a=min(
                        st.session_state.best_model, st.session_state.other_model
                    ),
                    model_b=max(
                        st.session_state.best_model, st.session_state.other_model
                    ),
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
                router.add_new_match(
                    prompt=st.session_state.current_prompt,
                    embedding=st.session_state.current_embedding,
                    model_a=min(
                        st.session_state.best_model, st.session_state.other_model
                    ),
                    model_b=max(
                        st.session_state.best_model, st.session_state.other_model
                    ),
                    score=2,  # Draw
                )
                st.success("✅ Feedback recorded as a draw! Router updated.")
                # Reset comparison state
                st.session_state.show_comparison = False
                st.session_state.response_other = None
                st.rerun()

    # Footer with current ELO scores
    if router.elo_scores:
        st.divider()
        st.subheader("📊 Current Model Rankings")
        col1, col2 = st.columns(2)
        for idx, (model, score) in enumerate(
            sorted(router.elo_scores.items(), key=lambda x: x[1], reverse=True)
        ):
            with col1 if idx % 2 == 0 else col2:
                st.metric(f"#{idx + 1} {model.split('-20')[0]}", f"ELO: {score:.1f}")


if __name__ == "__main__":
    main()
