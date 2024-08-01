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

def get_date():
    current_date = datetime.date.today()
    today = str(current_date).replace("-", "_")
    return today
