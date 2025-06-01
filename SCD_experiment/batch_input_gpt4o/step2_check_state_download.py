# @Time : 2024/9/25 10:54 

# @File : check_state.py 
# @Project: PyCharm
from openai import OpenAI
import os

os.environ['OPENAI_API_KEY'] = ""  # Assign your API keys accordingly

client = OpenAI()
state = client.batches.list(limit=10)  # 查看所有批处理
print(state)
"""
validating：正在验证输入文件，然后批处理才能开始
failed：输入文件未通过验证过程
in_progress：输入文件已成功验证，批处理正在运行
finalizing：批处理已完成，正在准备结果
completed：批处理已完成，结果已准备就绪
expired：批处理无法在 24 小时时间窗口内完成
cancelling：已启动批处理取消
cancelled：批处理已取消
"""
num = 6  # Set the number of files you want to process

for i, file in enumerate(state.data[0:num]):
    content = client.files.content(file.output_file_id).content
    name = file.metadata['description'][:-11] + "_output.json"
    with open(f"gci_batch_output/{name}", "wb") as f:
        f.write(content)
