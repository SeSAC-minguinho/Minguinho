from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    FewShotChatMessagePromptTemplate,
)
import json
from langchain_core.runnables import RunnableSerializable
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
import datetime

load_dotenv()


def load_model(model_name: str) -> ChatOpenAI:
    return ChatOpenAI(
        model_name=model_name,
        temperature=0.1,
        max_tokens=300,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def load_prompt(file_name: str) -> str:
    with open("{}.txt".format(file_name), "r", encoding="utf-8") as f:
        return f.read()


def load_fewshotprompt(file_name: str) -> FewShotChatMessagePromptTemplate:
    with open("{}_fewshot.txt".format(file_name), "r", encoding="utf-8") as f:
        qna = json.load(f)

    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{user}"), ("ai", "{ai}")]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt, examples=qna
    )
    return few_shot_prompt


def summarization(model: ChatOpenAI, chat_history):
    messages = [
        (
            "system",
            """
            너는 요약을 잘 하는 아이의 친구야. 다음은 아이와 AI가 나눈 대화 내용이야.
            대화 내용을 몇 줄로 요약해줘. 가능한 대화의 모든 정보를 빠짐 없이 포함시켜야해.
            아래 예시를 참고해서 작성해줘.
            
            예시 1: 
            아이는 집에서 친구들과 함께 미국과 세르비아의 농구 경기를 보고 있다. 
            AI는 농구 경기가 어떻게 진행되는지 질문하였으며, 아이는 미국이 8점차로 이기고 있다고 대답했다.
            AI가 어떤 농구 팀과 선수를 가장 좋아하냐는 질문에, 아이는 미국을 가장 좋아하고 르브론 제임스 선수를 제일 좋아한다고 대답하였다.
            AI가 또 어떤 운동을 좋아하냐고 묻자, 아이는 수영과 발레를 제일 좋아한다고 대답하였다.
            마지막으로, AI는 아이가 수영을 얼마나 잘하는지, 해달이 수영을 얼마나 좋아하고 잘 하는지 설명했다.
            """,
        ),
        ("human", chat_history),
    ]
    summary = model.invoke(messages).content
    return summary


def get_date():
    current_date = datetime.date.today()
    today = str(current_date).replace("-", "_")
    return today
