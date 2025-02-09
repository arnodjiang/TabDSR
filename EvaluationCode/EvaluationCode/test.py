import re

texts = [
    "打工。2018 年 9 月 23 日，余红武因形迹可疑，被公安机关审"
]

for text in texts:
    for match_p in re.finditer(r'\d{4}\s*年(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?)?', text):
        print(match_p.group())