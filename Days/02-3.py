from langchain_core.runnables import RunnableBranch, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("openai:gpt-5.4-nano")

technical_chain = (
    ChatPromptTemplate.from_messages(
        [   
            ("system", "당신은 시니어 개발자 멘토입니다. 질문을 코드와 구현 관점에서 단계별로 답하세요."),
            ("human", "{question}")
        ]
    )
    | model
    | StrOutputParser()
)

interview_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "당신은 면접 코치입니다. 다변의 구조와 개선 포인트를 짚어 주세요."),
        ("human", "{question}")
    ])
    | model
    | StrOutputParser()
)

general_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "당신은 친절한 학습 도우미입니다. 질문에 간결하고 정확하게 답하세요."),
        ("human", "{question}")
    ])
    | model
    | StrOutputParser()
)

TECHNICAL_KEYWORDS = ("코드", "구현")
INTERVIEW_KEYWORDS = ("면접", "자기소개")

def classify_question(x:dict) -> str:
    """질문 dict를 받아서 question_type 라벨 하나를 돌려주는 rule classifier"""
    question = x["question"]
    if any(k in question for k in TECHNICAL_KEYWORDS):
        return "technical"
    if any(k in question for k in INTERVIEW_KEYWORDS):
        return "interview"
    return "general"

def is_technical(x: dict) -> bool:
    print("[is_technical] 실행됨")
    return classify_question(x) == "technical"

def is_interview(x: dict) -> bool:
    print("[is_interview] 실행됨")
    return classify_question(x) == "interview"

# 위에서부터 평가, 처음 만나는 하나만 실행
branch = RunnableBranch(
    (is_technical, technical_chain),
    (is_interview, interview_chain),
    general_chain
)

q_technical = {
    "question" : "이 코드를 LCEL로 구현하려면 어디를 고치나요?"
    " 면접에서 자기소개 답변을 어디를 고치나요?"
}

q_interview = {
    "question": "면접에서 자기소개 답변을 어떻게 개선하면 좋나요?"
}

answer = branch.invoke(q_technical)
print(type(answer))
print(answer)