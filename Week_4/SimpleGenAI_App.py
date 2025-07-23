from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

load_dotenv()

llms = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7)

prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""
You are a helpful AI assistant.
User says: {user_input}
Your response:
"""
)

chain = LLMChain(llm=llms, prompt=prompt)

if __name__ == "__main__":
    user_input = input("Ask me anything: ")
    response = chain.run({"user_input": user_input})
    print("AI says:", response)
