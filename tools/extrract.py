###############################################################################

# AI Hub에서 상담 데이터를 추출하는 코드입니다.

###############################################################################

import os
import glob
import json

# Read Files
os.chdir(r'Path')
directory = r'Directory'
file_pattern = os.path.join(directory, '*.json')
files = glob.glob(file_pattern)

# Extract audio transcripts
questions = []
for file in files:
    with open(file, mode="r", encoding="utf-8") as f:
        data = json.load(f)
        for i in range(len(data['list'])):
            for j in range(len(data['list'][i]['list'])):
                if 'audio' in data['list'][i]['list'][j].keys():
                    for k in range(len(data['list'][i]['list'][j]['audio'])):
                        if data['list'][i]['list'][j]['audio'][k]['type'] == 'Q':
                            questions.append(data['list'][i]['list'][j]['audio'][k]['text'])

# Save transcripts into the text file
with open('questions.txt', 'w', encoding="utf-8") as f:
    for q in questions:
        f.write(f"{q}\n")
