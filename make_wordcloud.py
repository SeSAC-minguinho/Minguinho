##################################################

# 대락적인 wordcloud 코드입니다.

###################################################

from wordcloud import WordCloud
import matplotlib.pyplot as plt


# 텍스트 파일 읽기
with open("./Trabbit.txt", "r", encoding="utf-8") as file:
    text = file.read()


wordcloud = WordCloud(
    font_path="./YonseiBold.ttf",  # 한글 폰트 경로 필수로 필요함 
    width=800,  # 이미지의 너비
    height=400,  # 이미지의 높이
    max_words=100,  # 워드클라우드에 표시할 최대 단어 수
    background_color="white",  # 배경색
    colormap="rainbow"  # 단어 색상 맵
).generate(text)


# 워드클라우드 시각화
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")  # 축을 보이지 않게 설정
plt.show()
