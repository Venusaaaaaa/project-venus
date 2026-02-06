import streamlit as st
from agent import create_agent

st.set_page_config(
    page_title="Car Sensor Assistant",
    layout="centered"
)

st.title("🚗 Car Sensor Troubleshooting Assistant")
st.write("Ask anything about car sensors, MAP, MAF, O2, ECT, etc.")

# Initialize agent only once
if "agent" not in st.session_state:
    st.session_state.agent = create_agent()

# User input
user_query = st.text_input("Describe your problem:", "")

if st.button("Diagnose"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        st.write("⏳ Thinking...")
        result = st.session_state.agent.invoke({"input": user_query})
        final_answer = result["final_answer"]

        st.subheader("🔧 Diagnosis")
        st.write(final_answer)
