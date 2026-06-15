# LangChain을 사용할 때 4가지 대표 api
# model (init_chat_model) -> 프로바이더 별 모델 설정.
# prompt (ChatPromptTemplate) -> 프롬프트 구조화 또는 관리 기능
# parser (StrOutputParser) -> 출력 양식, 구조를 정하는 기능
# invoke() -> 파이프 라인 구성, 실행.

from langchain_core.prompts import ChatPromptTemplate

# role 기반 메시지 양식: system은 고정 지시문 (규칙), human(user)은 변수 칸 {topic} 포함
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 {topic} 기술 튜터입니다."),
    ("human", "{topic}을 초보자에게 3문장으로 설명해주세요.")
])
# filled = prompt.invoke({"topic" : "LCEL"})

# print(type(filled))
# print(filled.to_messages())

for t in ["LangChain", "Python", "LCEL"]:
    filled = prompt.invoke({"topic": t})
    print(filled.to_messages()[1].content)