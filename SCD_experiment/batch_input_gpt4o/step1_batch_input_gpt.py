# @Time : 2024/9/25 10:43

# @File : step1_batch_input.py
# @Project: PyCharm
from openai import OpenAI
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"

ref = [
    "F-E", "AP-DD", "E-MPF"]

client = OpenAI()
for type in ["llm4ele_qwen25-72b"]:
    for name in [0, 1, 2]:
        for file in [f"{type}/gcidata_total_{ref[name]}_batch_input.json"]:
            batch_input_file = client.files.create(
                file=open(file, "rb"),
                purpose="batch"
            )

            batch_input_file_id = batch_input_file.id

            client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": file
                }
            )
