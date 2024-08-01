
###############################################################################

##  github : https://github.com/cosmoquester/2021-dialogue-summary-competition 
##  전체 로그를 받아서 모델에 적용할 수 있는 코드가 필요합니다. - 성능은 기본적으로 좋습니다.

###############################################################################


from transformers import pipeline

model_name = "alaggung/bart-r3f"
max_length = 64


dialogue = ["밥 ㄱ?", "고고고고 뭐 먹을까?", "어제 김치찌개 먹어서 한식말고 딴 거", "그럼 돈까스 어때?", "오 좋다 1시 학관 앞으로 오셈", "ㅇㅋ"]


summarizer = pipeline("summarization", model=model_name)
summarization = summarizer("[BOS]" + "[SEP]".join(dialogue) + "[EOS]", max_length=max_length)

print(summarization)
