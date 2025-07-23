import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# ---------- Initialize LLM safely ----------
try:
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o",
        temperature=0.4
    )
except Exception as e:
    st.error(f"❌ Error initializing LLM: {e}")
    st.stop()

# ---------- Prompt ----------
prompt = PromptTemplate(
    input_variables=["role", "job_description"],
    template="""
You are a professional interview coach.

Given the following role and job description:

Role: {role}

Job Description:
{job_description}

Generate 5 realistic, targeted mock interview questions for this position. Questions should assess both technical and behavioral aspects relevant to the role.
"""
)

# ---------- Chain ----------
chain = LLMChain(llm=llm, prompt=prompt)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Mock Interview Generator", page_icon="🗂️", layout="centered")

st.markdown("<h1 style='text-align: center; color: #FF6F00;'>🗂️ Mock Interview Question Generator</h1>", unsafe_allow_html=True)
st.markdown("### Practice with **AI-generated, targeted mock interview questions** before your next interview 🚀")

# Job Role Input
role = st.text_input("💼 Enter the Role you are preparing for:", placeholder="e.g., Data Analyst")

# Job Description Input
job_description = st.text_area(
    "📄 Paste the Job Description / Responsibilities:",
    placeholder="Paste the JD to tailor the mock questions to your upcoming interview..."
)

st.markdown("---")

# Generate Mock Questions
if st.button("🎤 Generate Mock Questions"):
    if role.strip() == "" or job_description.strip() == "":
        st.warning("⚠️ Please enter both the role and the job description to generate questions.")
    else:
        with st.spinner("🎯 Generating your mock interview questions..."):
            try:
                response = chain.run({
                    "role": role,
                    "job_description": job_description
                })
                st.success("✅ Mock Interview Questions Generated Successfully!")
                st.subheader("🎤 Your Mock Interview Questions:")
                st.code(response, language='markdown')

                st.download_button(
                    label="💾 Download Questions as .txt",
                    data=response,
                    file_name=f"{role.replace(' ', '_')}_mock_interview_questions.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"❌ An error occurred while generating the mock questions: {e}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>✨ Built with ❤️ using LangChain + Streamlit ✨</p>", unsafe_allow_html=True)
