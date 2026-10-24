
import asyncio
import logging
import streamlit as st

from pinecone_rag import (
    initialize_index,
    search_vectors,
    generate_llm_answer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Spring Core Knowledge Base",
    page_icon="📘",
    layout="wide",
)

# --------------------------------------------------
# Custom styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }
    .hero {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(120deg, #17365D, #2563A6);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        color: white;
        margin-bottom: 0.5rem;
    }
    .hero p {
        color: #e5efff;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📘 Spring Core Knowledge Assistant</h1>
        <p>
            Ask questions about Spring Core and get answers
            grounded in your PDF documents using Pinecone RAG.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Initialize Pinecone index
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_index():
    return initialize_index()


# --------------------------------------------------
# Sidebar settings
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Search Settings")

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=10,
        value=5,
    )

    category_filter = st.selectbox(
        "Document category",
        options=["pdf-document", "All categories"],
        index=0,
    )

    show_context = st.checkbox(
        "Show retrieved context",
        value=True,
    )

    st.divider()
    st.caption("Powered by OpenAI + Pinecone")

# --------------------------------------------------
# Initialize the index
# --------------------------------------------------
try:
    with st.spinner("Connecting to Pinecone..."):
        index = get_index()

    st.sidebar.success("Pinecone connected")

except Exception as exc:
    logger.exception("Pinecone initialization failed")
    st.error(
        "Could not initialize Pinecone. Check your API keys, "
        "index configuration, and network connection."
    )
    st.exception(exc)
    st.stop()

# --------------------------------------------------
# Question input
# --------------------------------------------------
st.subheader("Ask a question")

with st.form("question_form"):
    question = st.text_area(
        "Enter your Spring Core question",
        placeholder=(
            "Example: Explain Spring transaction management "
            "and its best practices."
        ),
        height=100,
    )

    submitted = st.form_submit_button(
        "🔍 Search and Generate Answer",
        type="primary",
        use_container_width=True,
    )

# --------------------------------------------------
# Answer generation
# --------------------------------------------------
if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    try:
        with st.spinner(
            "Searching Pinecone and generating your answer..."
        ):
            # Retrieve relevant chunks from Pinecone
            results = asyncio.run(
                search_vectors(
                    index=index,
                    question=question.strip(),
                    category=(
                        None
                        if category_filter == "All categories"
                        else category_filter
                    ),
                    top_k=top_k,
                )
            )

            # Generate an answer using retrieved context
            answer = asyncio.run(
                generate_llm_answer(
                    question=question.strip(),
                    search_results=results,
                )
            )

        # ----------------------------------------------
        # Display answer
        # ----------------------------------------------
        st.divider()
        st.subheader("💡 Generated Answer")
        st.markdown(answer)

        # ----------------------------------------------
        # Display retrieved sources
        # ----------------------------------------------
        matches = results.matches

        st.divider()
        st.subheader("📚 Retrieved Sources")

        if not matches:
            st.info(
                "No matching chunks were returned from Pinecone."
            )
        else:
            st.caption(
                f"Retrieved {len(matches)} matching chunks."
            )

            for position, match in enumerate(matches, start=1):
                metadata = match.metadata or {}

                source = metadata.get("source", "Unknown source")
                chunk_index = metadata.get("chunk_index", "N/A")
                score = getattr(match, "score", None)
                text = metadata.get(
                    "text",
                    "No text available",
                )

                score_label = (
                    f"{score:.4f}"
                    if score is not None
                    else "N/A"
                )

                with st.expander(
                    f"Source {position}: {source} | "
                    f"Similarity: {score_label}"
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Document:** {source}")

                    with col2:
                        st.write(f"**Chunk index:** {chunk_index}")

                    st.write("**Retrieved text:**")
                    st.write(text)

        # ----------------------------------------------
        # Optional retrieved-context view
        # ----------------------------------------------
        if show_context and matches:
            with st.expander("View all retrieved context"):
                for position, match in enumerate(matches, start=1):
                    metadata = match.metadata or {}

                    st.markdown(
                        f"**Context {position} — "
                        f"{metadata.get('source', 'Unknown')}**"
                    )
                    st.write(metadata.get("text", ""))
                    st.divider()

    except Exception as exc:
        logger.exception("Question processing failed")
        st.error(
            "Unable to process your question. Check the "
            "OpenAI API key, Pinecone connectivity, and logs."
        )
        st.exception(exc)

# --------------------------------------------------
# Pinecone index statistics
# --------------------------------------------------
st.divider()

with st.expander("📊 Pinecone Index Statistics"):
    try:
        stats = index.describe_index_stats()

        st.json(
            stats.to_dict()
            if hasattr(stats, "to_dict")
            else dict(stats)
        )

    except Exception:
        logger.exception("Could not retrieve index statistics")
        st.warning("Index statistics are currently unavailable.")

st.caption(
    "Answers are generated from retrieved document content. "
    "Verify important technical decisions against the original documentation."
)
