##################################################

# Word Cloud 생성 코드입니다.
# 아이와 AI 대화 기록을 매개변수로 받아옵니다.
# 아이의 대화기록만 Cloud로 생성합니다.

###################################################

from wordcloud import WordCloud
import matplotlib.pyplot as plt


def generate_cloud(text: str) -> None:
    wordcloud = WordCloud(
        font_path="./YonseiBold.ttf",  # 한글 폰트 경로 필수로 필요함
        width=800,  # 이미지의 너비
        height=400,  # 이미지의 높이
        max_words=100,  # 워드클라우드에 표시할 최대 단어 수
        background_color="white",  # 배경색
        colormap="rainbow",  # 단어 색상 맵
        collocations=False
    ).generate(text)

    # 워드클라우드 시각화
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")  # 축을 보이지 않게 설정
    plt.show()
