import os
import streamlit as st
from dotenv import load_dotenv
from rag import RAGCore
from web_corrective import CorrectiveSearcher

# Load the env variables
load_dotenv()

# Setup page title
st.set_page_config(page_title="Seora - Startup Pitch AI Agent", page_icon="📊", layout="wide")

# Setup RAG Agent if it's not in the session
if "rag" not in st.session_state:
    st.session_state.rag = RAGCore(
        chroma_dir=os.getenv("CHROMA_DIR", ".chroma"),
        collection_name=os.getenv("COLLECTION_NAME", "startup_mvp"),
        embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small")
    )

# Setup the Web Corrective Searcher if it's not in the session
if "corrective" not in st.session_state:
    st.session_state.corrective = CorrectiveSearcher()

# Upload key as a gate to ensure after clearing database, it doesn't upload automatically
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

rag = st.session_state.rag
corrective = st.session_state.corrective

# Streamlit Title and tabs UI
st.title("📊 Seora - Startup Pitch AI Agent")
tab_upload, tab_chat = st.tabs(["📤 Upload", "💬 Chat"])

# Upload Section of the website
with tab_upload:
    # Setup file uploader
    st.subheader("Upload startup materials")
    uploaded = st.file_uploader(
        "Upload documents", type=["pdf", "pptx", "csv", "docx", "png", "jpg", "jpeg", "tiff", "tif"], 
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.upload_key}",
        help="Supported formats: PDF, PowerPoint, Excel, Word documents, and images (PNG, JPG, TIFF). Images will be processed with OCR to extract text."
    )

    # If a file has been uploaded, for every file, we give it a spinner icon for loading and index it into the vectore store for the RAG agent
    if uploaded:
        for f in uploaded:
            with st.spinner(f"Indexing {f.name}..."):
                rag.index_file(f, filename=f.name)
        st.success("Files indexed!")

    # Clear Database button
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑️ Clear Database", type="secondary", use_container_width=True):
            rag.reset()                            # Clear the RAG database
            st.session_state.upload_key += 1       # Increment the upload key to force a fresh file uploader
            
            # Clear chat history
            if "history" in st.session_state:
                st.session_state.history = []
            st.success("Database cleared! All files and indexes removed.")
            st.rerun()


# Chat section of the website
with tab_chat:
    st.subheader("Chat with your corpus")
    user_q = st.text_input("Ask a question", key="question_input")
    
    # Settings area with a slider for context window (4 by default)
    st.write("**Settings:**")
    top_k = st.slider("Context Window", 1, 10, 4, 
                      help="Number of document chunks to consider when answering (higher = more comprehensive, lower = more focused)")
    
    # Advanced settings in an expander
    with st.expander("⚙️ Advanced Settings"):
        do_corrective = st.checkbox("Enable Web-Searching Correction", value=True,
                                   help="When enabled, automatically searches the web to enhance answers and fill knowledge gaps. Disable if you prefer answers only from uploaded documents.")

    # Setting history and reset_chats flags if not in session state
    if "history" not in st.session_state:
        st.session_state.history = []
    if "reset_flag" not in st.session_state:
        st.session_state.reset_flag = False

    # Reset Chat button - placed BEFORE question processing
    if st.button("Reset chat"):
        # Only reset if there's history to reset
        if "history" in st.session_state and st.session_state.history:
            st.session_state.history = []
            st.session_state.reset_flag = True  # Set flag to prevent processing
            st.success("Chat reset.")
            st.rerun()
        else:
            st.info("💬 No chat history to reset.")

    # This is when a user enters a query
    if user_q and user_q.strip() and not st.session_state.reset_flag:
        # If entered an emtpy string
        if not user_q.strip():
            st.warning("Enter a question.")
        
        else:
            # Check if database has any documents first
            try:
                # Try to get a simple count or check if database is empty
                with st.spinner("Checking database..."):
                    # This will fail if database is empty, triggering the web search fallback
                    result = rag.answer(user_q, top_k=top_k)
                
                # If documents and context found it will continue as such:
                st.session_state.history.append(("user", user_q))
                content = result["answer"]
                contexts = result["contexts"]
                coverage = result["coverage"]
                sources = result["sources"]
                
                # If the corrective feature is on, and coverage is less than 0.5 (not enough need to search)
                if do_corrective and coverage < 0.5 and corrective.enabled:
                    st.info(f"🔄 Coverage low: {coverage:.2f} — Searching the web to enhance your answer...")

                    # Will execute the web search here
                    with st.spinner("Running web search..."):
                        web_ctx, web_sources = corrective.search(user_q, contexts)

                        # If received web context then it will merge the context from the documents and websearch and input to the RAG agent
                        if web_ctx:
                            merged_ctx = contexts + web_ctx
                            result2 = rag.answer(user_q, top_k=top_k, override_contexts=merged_ctx)
                            content = result2["answer"]
                            contexts = merged_ctx
                            sources = list(set(sources + web_sources))
                            coverage = result2["coverage"]
                            st.success(f"✅ Web search completed! Found {len(web_ctx)} additional sources.")
                        else:
                            st.warning("⚠️ Web search completed but no additional sources found.")
                
                # If coverage isn't enough and web search is disabled 
                else:
                    st.info("ℹ️ Coverage sufficient — upload more documents or try enabling web search.")

                # Display the user's and assistant's messages
                st.session_state.history.append(("assistant", content))
                for role, msg in st.session_state.history[-6:]:
                    st.chat_message("user" if role == "user" else "assistant").markdown(msg)

                # Expander for contexts and sources
                with st.expander("📎 Context & Sources"):
                    for i, ctx in enumerate(contexts):
                        meta = ctx.get("metadata", {})
                        st.markdown(f"**{i+1}. {meta.get('source', 'snippet')}**\n{ctx.get('text','')[:600]}...")
                    if sources:
                        st.markdown("**External sources:**")
                        for s in sources:
                            st.markdown(f"- {s}")
            
            # If there is nothing in the database, this exception occurs - web searching
            except Exception as e:
                st.info("📚 No documents found — searching the web for your question...")
                
                # If the corrective search features are enabled
                if do_corrective and corrective.enabled:
                    with st.spinner("Searching the web..."):

                        # Searching the user's query though the class function
                        web_ctx, web_sources = corrective.search(user_q, [])
                        
                        # If there is web context results, we will form a start-up focused answer: 
                        if web_ctx:
                            content = "**Startup Analysis Based on Web Research:**\n\n"
                            for i, ctx in enumerate(web_ctx[:3]):
                                content += f"**{i+1}. Key Insight:** {ctx['text'][:300]}...\n\n"
                            content += "**Strategic Considerations:**\n"
                            content += "- Market opportunity assessment\n"
                            content += "- Competitive landscape analysis\n"
                            content += "- Business model validation\n\n"
                            content += f"**Sources:** {', '.join(web_sources[:3])}"
                            
                            st.success(f"✅ Web search completed! Found {len(web_ctx)} sources.")
                            
                            # Display user query and the assistant's response
                            st.session_state.history.append(("user", user_q))
                            st.session_state.history.append(("assistant", content))
                            for role, msg in st.session_state.history[-6:]:
                                st.chat_message("user" if role == "user" else "assistant").markdown(msg)
                            
                            # Expander for web sources
                            with st.expander("📎 Web Sources"):
                                for s in web_sources[:3]:
                                    st.markdown(f"- {s}")
                        else:
                            st.error("❌ Web search failed. Please try again later.")
                else:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 **Tip:** Enable web search or upload some documents first.")




