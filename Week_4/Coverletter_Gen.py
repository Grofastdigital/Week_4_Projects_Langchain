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
        temperature=0.5
    )
except Exception as e:
    st.error(f"❌ Error initializing LLM: {e}")
    st.stop()

# ---------- Prompt ----------
prompt = PromptTemplate(
    input_variables=["job_title", "experience_level", "tone", "job_description", "resume_text"],
    template="""
You are a professional career coach and expert cover letter writer.

Using the following candidate resume content:
{resume_text}

Write a compelling, clean, ATS-friendly cover letter for the following details:

- **Job Title:** {job_title}
- **Experience Level:** {experience_level}
- **Tone:** {tone}
- **Job Description / Keywords:** {job_description}

The cover letter should highlight relevant skills from the resume, match the job description, and be ready to send with a professional closing.
"""
)

# ---------- Chain ----------
chain = LLMChain(llm=llm, prompt=prompt)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Smart Cover Letter Builder", page_icon="📝", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📝 Smart Cover Letter Builder</h1>", unsafe_allow_html=True)
st.markdown("### Generate a **tailored, ATS-friendly cover letter** using your resume in seconds 🚀")

# Upload Resume
resume_file = st.file_uploader("📄 Upload Your Resume (PDF or TXT):", type=["pdf", "txt"])

# Job Title Input
job_title = st.text_input("💼 Job Title (Position Applying For):", placeholder="e.g., Marketing Manager")

# Experience Level
experience_level = st.selectbox(
    "🎯 Experience Level:",
    ["Entry Level", "Mid Level", "Senior Level", "Internship", "Career Change"]
)

# Tone
tone = st.selectbox(
    "🎨 Tone of the Cover Letter:",
    ["Formal", "Friendly", "Confident", "Persuasive"]
)

# Job Description
job_description = st.text_area(
    "📄 Job Description / Keywords:",
    placeholder="Paste the job description or keywords you want to highlight..."
)

st.markdown("---")

# Extract text from uploaded resume
def extract_text_from_resume(uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".txt"):
                return uploaded_file.read().decode("utf-8")
            elif uploaded_file.name.endswith(".pdf"):
                from PyPDF2 import PdfReader
                reader = PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
            else:
                return ""
        except Exception as e:
            st.error(f"❌ Error reading the resume file: {e}")
            return ""
    return ""

# Generate Cover Letter
if st.button("🚀 Generate Cover Letter"):
    if job_title.strip() == "" or job_description.strip() == "" or resume_file is None:
        st.warning("⚠️ Please upload your resume and fill in the job title and job description.")
    else:
        resume_text = extract_text_from_resume(resume_file)
        if resume_text.strip() == "":
            st.warning("⚠️ Could not extract text from the uploaded resume. Please check your file.")
        else:
            with st.spinner("✍️ Generating your cover letter..."):
                try:
                    response = chain.run({
                        "job_title": job_title,
                        "experience_level": experience_level,
                        "tone": tone,
                        "job_description": job_description,
                        "resume_text": resume_text
                    })
                    st.success("✅ Cover Letter Generated Successfully!")
                    st.subheader("📄 Your Cover Letter:")
                    st.code(response, language='markdown')

                    st.download_button(
                        label="💾 Download Cover Letter as .txt",
                        data=response,
                        file_name=f"{job_title.replace(' ', '_')}_cover_letter.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"❌ An error occurred while generating the cover letter: {e}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>✨ Built with ❤️ using LangChain + Streamlit ✨</p>", unsafe_allow_html=True)
