from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("openai:gpt-5.4-nano")

def primary_fn(x:dict )->dict:
    """주 경로, force_fail이 참이면 항상 같은 예외 발생"""
    if x.get("force_fail"):
        raise ValueError("mock primary failure")
    return {
        "answer": f"primary: {x['question']}",
        "meta": {"fallback_used": False, "error_type": None, "error_stage": None, "route_type": x.get("route_type")}
    }

primary = RunnableLambda(primary_fn)

# 정상 경로 - 예외가 없는 경우
print(primary.invoke({"question": "정상 경로 확인", "route_type": "branch:faq"}))

# 실패 경로 - 예외가 있는 , fallback이 없다. 예외가 그대로 노출
primary.invoke({"question": "실패 경로 확인", "route_type": "branch:faq", "force_fail": True})