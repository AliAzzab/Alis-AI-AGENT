import streamlit as st
from agent_logic import agent

st.set_page_config(page_title="Groq AI Research Agent", page_icon="🧠")
st.title("🧠 Groq AI Research Assistant")

query = st.text_input("🔎 Ask me anything:")

if query:
    with st.spinner("Thinking..."):
        try:
            response = agent.run(query)
            st.success("Done!")
            st.markdown("### 📎 Response")
            st.write(response)

            # Option to download answer
            st.download_button("💾 Download", response, file_name="response.txt")

        except Exception as e:
            st.error(f"❌ Error: {e}")