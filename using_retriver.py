from langchain_openai import OpenAIEmbeddings  # solar임베딩도 있음
from langchain.vectorstores import Chroma  # langchain에서 불러옴
from dotenv import load_dotenv

#.env key값 읽어오기
load_dotenv()

# Chroma 데이터베이스 불러오기
persist_directory = "./chroma_db"

# embedding_function 설정
embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

# 변수명에 직접 적어놓으니 오류가 발생함 위에서 각각 정의하고 입력할 것
vectorstore = Chroma(
    persist_directory=persist_directory, embedding_function=embedding_function
)

query = "나는 친구가 없다. 그래서 말할 사람도 없다. 엄마아빠는 나를 신경쓰지도 않는다."
docs = vectorstore.similarity_search_with_relevance_scores(query, k=20)
print(*(docs[i] for i in range(len(docs)) ))




# # result =vectorstore.similarity_search(query)

# # print("가장 유사한 문서:\n {}\n".format(docs[0][å0].page_content))
# # print("문서 유사도: \n {}".format(docs[0][1]))




# # 검색 쿼리
# query = "친구와 게임을 하다가 순서 때문에 싸웠다."

# # 가장 유사도가 높은 문장 3개 추출
# retriever = vectorstore.as_retriever(score_threshold = 0.8,search_kwargs={"k": 5})

# # 관련 문장 저장
# docs = retriever.invoke(query)
# # docs = retriever.get_relevant_documents(query) 위와 동일하나 곧 사라집니다. invoke를 사용하세요

# # 결과값 출력 예시
# print(len(docs))
# print(docs)
