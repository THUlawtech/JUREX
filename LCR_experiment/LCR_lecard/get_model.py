# @Time : 2024/8/24 15:33 

# @File : get_model.py 
# @Project: PyCharm

#模型下载
from modelscope import snapshot_download

model_dir = snapshot_download('Xorbits/bge-m3',cache_dir='/root/autodl-tmp')

# model_dir = snapshot_download('PollyZhao/bert-base-chinese',cache_dir='/root/autodl-tmp')
# model_dir = snapshot_download('Qwen/Qwen2.5-72B-Instruct')