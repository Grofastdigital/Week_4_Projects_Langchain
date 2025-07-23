import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------- LLM Initialization ----------------
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0.4
)

# ---------------- Prompt Template ----------------
prompt = PromptTemplate(
    input_variables=["email_task", "tone", "subject_toggle"],
    template="""
You are a professional email writing assistant. Your task is to generate a clear, polite, and effective email.

Email Purpose:
{email_task}

Tone:
{tone}

{subject_toggle}

Generate a professional, ready-to-send email.
"""
)

# ---------------- LLM Chain ----------------
chain = LLMChain(llm=llm, prompt=prompt)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Smart Email Writer", page_icon="📧", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📧 Smart Email Writer</h1>", unsafe_allow_html=True)
st.markdown("### Draft professional emails in seconds using AI 🚀")

email_task = st.text_area("🖊️ What is your email about?", placeholder="e.g., Follow up with a client for feedback.")

col1, col2 = st.columns(2)

with col1:
    tone = st.selectbox("🎨 Select Email Tone:", ["Friendly", "Formal", "Persuasive"])

with col2:
    subject_toggle = st.checkbox("✨ Generate Subject Line", value=True)

st.markdown("---")

if st.button("🚀 Generate Email"):
    if email_task.strip() == "":
        st.warning("⚠️ Please enter the email context.")
    else:
        with st.spinner("✍️ Generating your email, please wait..."):
            subject_instruction = "Include a clear, catchy subject line." if subject_toggle else "Do not include a subject line."
            try:
                response = chain.invoke({
                    "email_task": email_task,
                    "tone": tone,
                    "subject_toggle": subject_instruction
                })["text"]
                st.success("✅ Email Generated Successfully!")
                st.subheader("✉️ Your Email Draft:")
                st.code(response, language='markdown')

                st.download_button(
                    label="💾 Download Email as .txt",
                    data=response,
                    file_name="generated_email.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"❌ Error generating email: {e}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Built with ❤️ using LangChain + Streamlit</p>", unsafe_allow_html=True)
