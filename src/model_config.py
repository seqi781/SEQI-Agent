import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()

os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "agent")

llm = ChatOpenAI(
    model="gpt-5.4-nano",
)
