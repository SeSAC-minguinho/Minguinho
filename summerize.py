###############################################################################
# 아이와 AI 대화를 요약하는 함수
###############################################################################
from chat_utils import load_model
from langchain_core.prompts import ChatPromptTemplate
import os

with open("prompts/Summarization_system_prompt.txt", "r") as f:
    summarization_system_prompt = f.read()

summarization_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", summarization_system_prompt),
        ("human", "{input}")
    ]
)

def daily_report(text, model=load_model("gpt-4o")):
    chain = summarization_prompt | model
    return chain.invoke(text).content
