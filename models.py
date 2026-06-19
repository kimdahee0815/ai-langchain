from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

DEFAULT_MODEL = "openai:gpt-5.4-nano"

def get_model(model_id: str = DEFAULT_MODEL):
    return init_chat_model(model_id)

