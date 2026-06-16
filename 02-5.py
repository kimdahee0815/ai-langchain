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

def fallback_fn(x:dict) -> dict:
    """예외 처리를 위한 대체 경로. 항상 같은 답을 돌려주는 deterministic mock"""
    # exception_key="error" -> 처리된 예외 "객체"가 x["error"]로 주입된다.
    # 이 주입이 성립하려면 primary와 fallback 모두 입력 dict여야 한다.
    err = x.get("error")
    return {
        "answer": "deterministic mock fallback answer",
        "meta":{
            "fallback_used": True,
            "error_type": type(err).__name__ if err else None, # 타입 이름만 기록
            "error_stage": "primary_runnable",
            "route_type": "fallback:mock" # branch:* 와 분리된 fallback
        }
    }
    
    
primary = RunnableLambda(primary_fn).with_fallbacks(
    [RunnableLambda(fallback_fn)], 
    exceptions_to_handle=(ValueError, ),
    exception_key="error")

# 정상 경로 - 예외가 없는 경우
print(primary.invoke({"question": "정상 경로 확인", "route_type": "branch:faq"}))

# 실패 경로 - 예외가 있는 , fallback이 없다. 예외가 그대로 노출
# primary.invoke({"question": "실패 경로 확인", "route_type": "branch:faq", "force_fail": True})