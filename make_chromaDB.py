from langchain_openai import OpenAIEmbeddings  # solar임베딩도 있음
from langchain.vectorstores import Chroma  # langchain에서 불러옴
from langchain_text_splitters import CharacterTextSplitter
import os
from dotenv import load_dotenv

#.env key값 읽어오기
load_dotenv()

# 파일의 내용을 읽어서 file 변수에 저장합니다. #readline은 각 줄로 읽음
with open("./questions.txt") as f:
    file = f.read()


# 문서를 청크로 분할합니다.
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=30,
    chunk_overlap=5,
    length_function=len,
)
docs = text_splitter.split_text(file)

print("split 완료 완료!")


n_doc = len(docs)
i, j = 0, 10000
delta = 10000

while i < n_doc:
    if j > n_doc:
        j = n_doc
    vectorstore = Chroma.from_texts(
        docs[i:j],
        OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_directory="./chroma_db",
    )
    print(f"Saved documents {i + 1} to {j}")
    i = j
    j += delta

vectorstore.persist()


def main():
    print("이 코드는 현재 파일에서만 실행됩니다.")

if __name__ == "__main__":
    main()