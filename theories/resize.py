import re

file_name = '1.模型知识.md'
with open(file_name, 'r', encoding='utf-8') as f:
    txt = f.read()
    # content = re.sub('style="zoom:\d+%;"', 'width="50%" height="50%"', txt)
    content = re.sub('width:50% height=50% ', 'width="50%" height="50%"', txt)

with open(file_name, 'w', encoding='utf-8') as f:
    f.write(content)
