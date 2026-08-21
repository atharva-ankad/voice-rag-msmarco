import streamlit as st
import requests

# UPDATE THIS TO YOUR LIVE MODAL URL LATER
BACKEND_URL = "http://localhost:8000/chat/audio"

# Configure the page
st.set_page_config(page_title="Voice RAG", page_icon="🎙️", layout="centered")

st.title("🎙️ Voice-Enabled RAG Pipeline")
st.markdown("**Task 2: MSMARCO-XI Multilingual Query System**")
st.markdown("---")

# 1. Native Streamlit audio recorder 
audio_bytes = st.audio_input("1. Speak your query")

if audio_bytes is not None:
    # 2. Send to backend immediately when recorded
    try:
        # Streamlit handles the audio as a BytesIO buffer, perfectly ready for requests
        files = {"audio": ("query.wav", audio_bytes, "audio/wav")}
        
        with st.spinner("Processing pipeline (~5 seconds)..."):
            response = requests.post(BACKEND_URL, files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            st.subheader("2. Generated Answer")
            # Handle Guardrail blocks gracefully
            if not data.get("safe"):
                st.error(f"🛡️ **BLOCKED by Guardrails:** {data.get('error', 'Unknown safety error')}")
                st.write(data.get("response", ""))
            else:
                st.success("✅ Query processed safely.")
                st.write(data.get("response", "No response generated."))
            
            # Show the raw JSON for the judges
            st.subheader("3. Backend Execution Payload")
            st.json(data)
            
        else:
            st.error(f"❌ Server Error: {response.status_code}")
            st.write(response.text)
            
    except requests.exceptions.RequestException as e:
        st.error(f"🔌 Connection failed. Is the FastAPI backend running? Error: {str(e)}")