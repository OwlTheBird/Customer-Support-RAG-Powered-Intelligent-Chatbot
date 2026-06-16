from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import AI_API_KEY, PROMPT_TEMPLATE

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",  # other options (gemini-3-flash-preview | gemini-3.1-pro-preview)
    google_api_key=AI_API_KEY,
)

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."), ("user", PROMPT_TEMPLATE)]
)

chain = prompt | llm


def extract_text(content) -> str:
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return content
