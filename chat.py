
################################################

 # FastAPI와 연결되는 import 필요 

################################################


from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
import chat_utils
from langchain_community.chat_message_histories import RedisChatMessageHistory
from make_wordcloud import generate_cloud

# InMemoryChatMessageHistory를 사용할 경우 아래 코드를 사용
# store = {}
# def get_session_history(session_id):
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#     return store[session_id]

# 3가지 페르소나 중 하나를 선택

no_persona = input(
    """"Choose AI Persona:
1: Haeyong
2: Trabiit 
3: Kkabuk
"""
)

if no_persona == "1":
    persona = "Haeyong"
elif no_persona == "2":
    persona = "Trabbit"
elif no_persona == "3":
    persona = "Kkabuk"
else:
    raise ("Value Error!")

# 선택한 페르소나의 프롬프트 로딩
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            chat_utils.load_prompt(persona),
        ),
        # FewShotPrompt
        chat_utils.load_fewshotprompt(persona),
        ("human", "{input}"),
    ]
)

# 모델 불러오기
model = chat_utils.load_model("gpt-4o")

# 프롬프트와 모델로 체인 생성
chain = prompt | model

# With Redis 챗봇 생성
with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: RedisChatMessageHistory(
        session_id, url="redis://localhost:6379"
    ),
    history_messages_key="output",
)

# 챗봇 생성
# InMemoryChatMessageHistory를 사용할 경우 아래 코드를 사용
# with_message_history = RunnableWithMessageHistory(
#     chain,
#     get_session_history,
#     history_messages_key="output"
# )

today_persona = chat_utils.get_date() + "_" + persona

# 세션 아이디 'today_persona'에 저장 (today_persona: 2024_08_01_Haeyong)
config = {"configurable": {"session_id": today_persona}}


########################################################################

# docker 에서 redis 구현하고 연결하는 코드가 필요합니다


########################################################################3


# 이미 저장된 Redis 읽어오기
redis_history = RedisChatMessageHistory(today_persona, url="redis://localhost:6379")
print(redis_history)

# Redis 메시지 초기화 하고 싶을 때 사용
# redis_history.clear()
# print(redis_history)


# 대화가 시작되면 페르소나가 아동에게 말을 겁니다.
if not redis_history:
    if persona == "Haeyong":
        redis_history.add_ai_message(
            "안녕! 나는 {}이야! 오늘은 어떻게 재밌게 놀아볼까?".format("해용")
        )
    if persona == "Trabbit":
        redis_history.add_ai_message(
            "안녕! 나는 {}이야! 오늘은 어떻게 재밌게 놀아볼까?".format("트래빗")
        )
    if persona == "Kkabuk":
        redis_history.add_ai_message(
            "안녕! 나는 {}이야! 오늘은 어떻게 재밌게 놀아볼까?".format("까붓")
        )

    print(redis_history)


child_message = ""
for dialogue in redis_history.messages:
    if isinstance(dialogue, HumanMessage):
        child_message += dialogue.content
        child_message += "\n"
generate_cloud(child_message)

# 챗봇 시작
while True:
    ####################################################
     # input을 아이의 입력에 따라 챗봇이 진행될 수 있도록 하는 코드가 필요합니다.
    ####################################################
    
    query = input("Type your query: ")

    # AI 답변 생성
    result = with_message_history.invoke(input={"input": query}, config=config)

    print(result.content)

    # 버튼 누르면 끝내는 코드 작성 필요
