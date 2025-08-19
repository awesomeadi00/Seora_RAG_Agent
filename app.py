import os
import streamlit as st
from dotenv import load_dotenv
from rag import RAGCore
from web_corrective import CorrectiveSearcher

load_dotenv()

st.set_page_config(page_title="Seora - Startup Pitch AI Agent", page_icon="📊", layout="wide")

if "rag" not in st.session_state:
    st.session_state.rag = RAGCore(
        chroma_dir=os.getenv("CHROMA_DIR", ".chroma"),
        collection_name=os.getenv("COLLECTION_NAME", "startup_mvp"),
        embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small")
    )
if "corrective" not in st.session_state:
    st.session_state.corrective = CorrectiveSearcher()
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0


rag = st.session_state.rag
corrective = st.session_state.corrective

st.title("📊 Seora - Startup Pitch AI Agent")

tab_upload, tab_chat = st.tabs(["📤 Upload", "💬 Chat"])

with tab_upload:
    st.subheader("Upload startup materials")
    uploaded = st.file_uploader(
        "Upload documents", type=["pdf", "pptx", "csv", "docx", "png", "jpg", "jpeg", "tiff", "tif"], 
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.upload_key}",
        help="Supported formats: PDF, PowerPoint, Excel, Word documents, and images (PNG, JPG, TIFF). Images will be processed with OCR to extract text."
    )
    if uploaded:
        for f in uploaded:
            with st.spinner(f"Indexing {f.name}..."):
                rag.index_file(f, filename=f.name)
        st.success("Files indexed!")

    # Clear Database button
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑️ Clear Database", type="secondary", use_container_width=True):
            # Clear the RAG database
            rag.reset()
            # Increment the upload key to force a fresh file uploader
            st.session_state.upload_key += 1
            # Clear chat history
            if "history" in st.session_state:
                st.session_state.history = []
            st.success("Database cleared! All files and indexes removed.")
            st.rerun()



with tab_chat:
    st.subheader("Chat with your corpus")
    
    user_q = st.text_input("Ask a question", key="question_input")
    
    st.write("**Settings:**")
    top_k = st.slider("Context Window", 1, 10, 4, 
                      help="Number of document chunks to consider when answering (higher = more comprehensive, lower = more focused)")
    
    # Advanced settings in an expander
    with st.expander("⚙️ Advanced Settings"):
        do_corrective = st.checkbox("Enable Web-Searching Correction", value=True,
                                   help="When enabled, automatically searches the web to enhance answers and fill knowledge gaps. Disable if you prefer answers only from uploaded documents.")

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

    # Handle Enter key press - but only if not just reset
    if user_q and user_q.strip() and not st.session_state.reset_flag:
        if not user_q.strip():
            st.warning("Enter a question.")
        else:
            # Check if database has any documents first
            try:
                # Try to get a simple count or check if database is empty
                with st.spinner("Checking database..."):
                    # This will fail if database is empty, triggering the web search fallback
                    result = rag.answer(user_q, top_k=top_k)
                
                st.session_state.history.append(("user", user_q))

                content = result["answer"]
                contexts = result["contexts"]
                coverage = result["coverage"]
                sources = result["sources"]
                
                # Debug web corrective
                # st.sidebar.write("**Debug Info:**")
                # st.sidebar.write(f"Coverage: {coverage:.2f}")
                # st.sidebar.write(f"Corrective enabled: {corrective.enabled}")
                # st.sidebar.write(f"Do corrective: {do_corrective}")
                
                if do_corrective and coverage < 0.5 and corrective.enabled:
                    st.info(f"🔄 Coverage low: {coverage:.2f} — Searching the web to enhance your answer...")
                    with st.spinner("Running web search..."):
                        web_ctx, web_sources = corrective.search(user_q, contexts)
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
                else:
                    st.info("ℹ️ Coverage sufficient — using your uploaded documents.")

                st.session_state.history.append(("assistant", content))

                for role, msg in st.session_state.history[-6:]:
                    st.chat_message("user" if role == "user" else "assistant").markdown(msg)

                with st.expander("📎 Context & Sources"):
                    for i, ctx in enumerate(contexts):
                        meta = ctx.get("metadata", {})
                        st.markdown(f"**{i+1}. {meta.get('source', 'snippet')}**\n{ctx.get('text','')[:600]}...")
                    if sources:
                        st.markdown("**External sources:**")
                        for s in sources:
                            st.markdown(f"- {s}")
                
            except Exception as e:
                # If database is empty or any error, try web search instead
                st.info("📚 No documents found — searching the web for your question...")
                
                if do_corrective and corrective.enabled:
                    with st.spinner("Searching the web..."):
                        web_ctx, web_sources = corrective.search(user_q, [])
                        if web_ctx:
                            # Create a startup-focused answer from web results
                            content = "**Startup Analysis Based on Web Research:**\n\n"
                            for i, ctx in enumerate(web_ctx[:3]):
                                content += f"**{i+1}. Key Insight:** {ctx['text'][:300]}...\n\n"
                            content += "**Strategic Considerations:**\n"
                            content += "- Market opportunity assessment\n"
                            content += "- Competitive landscape analysis\n"
                            content += "- Business model validation\n\n"
                            content += f"**Sources:** {', '.join(web_sources[:3])}"
                            
                            st.success(f"✅ Web search completed! Found {len(web_ctx)} sources.")
                            
                            st.session_state.history.append(("user", user_q))
                            st.session_state.history.append(("assistant", content))
                            
                            for role, msg in st.session_state.history[-6:]:
                                st.chat_message("user" if role == "user" else "assistant").markdown(msg)
                            
                            with st.expander("📎 Web Sources"):
                                for s in web_sources[:3]:
                                    st.markdown(f"- {s}")
                        else:
                            st.error("❌ Web search failed. Please try again later.")
                else:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 **Tip:** Enable web search or upload some documents first.")




