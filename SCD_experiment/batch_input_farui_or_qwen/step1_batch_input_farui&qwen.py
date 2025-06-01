import json, asyncio, logging, time, math, sys, re, os, subprocess, traceback, hashlib, argparse

from sympy.physics.units import temperature
from tqdm import tqdm
from multiprocessing import Pool
from http import HTTPStatus
import dashscope
import os
import json

parser = argparse.ArgumentParser()

parser.add_argument('--log_file', type=str, default='./logs/error_v1.log')
parser.add_argument('--model_name', type=str, default='farui-plus')
parser.add_argument('--read_file_name', type=str,
                    default="/etc/ssd1/general_sft/evaluation/RewardModel/Judgement/rm_cross_validation_vote_v2_random_preds.jsonl")
parser.add_argument('--write_file_name', type=str,
                    default="/etc/ssd1/general_sft/evaluation/RewardModel/Judgement/rm_cross_validation_vote_v2_judge_v0.jsonl")
# parser.add_argument('--prompt_type', type=str, default="v0", choices=['v0', 'v3'])
parser.add_argument('--process_per_split', type=int, default=100)
parser.add_argument('--process_num', type=int, default=30)
parser.add_argument('--data_num', type=int, default=100)


args = parser.parse_args()

# 用process_num个进程处理, 每个进程处理process_per_split个数据
process_per_split = args.process_per_split
process_num = args.process_num

read_file_name = args.read_file_name
write_file_name = args.write_file_name


# 记录运行情况的文件
def set_logger(filename, filepath):
    os.makedirs(filepath, exist_ok=True)

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f'{filepath}/{filename}.log')
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s',
                                  datefmt='%a, %d %b %Y %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


logger = set_logger('info', '../../')


#### 以下代码是为了判断query.json与output.json的差集，获得还未调用openAI的数据
def dict_without_key(d, keys):
    return {k: v for k, v in d.items() if k not in keys}


# 字典转为hash
def hash_dict(d):
    # 排序以保证不受key顺序影响
    hash_object = hashlib.sha3_512(json.dumps(d, ensure_ascii=False, sort_keys=True).encode('utf-8'))
    return hash_object.hexdigest()


# 获取需要处理的数据
f_read = open(read_file_name, 'r')
if not os.path.exists(write_file_name):
    open(write_file_name, 'w').close()  # 若没有则创建
f_write_to_read = open(write_file_name, 'r')

f_all = []
f_now = []
for line in tqdm(f_read):
    # print(line)
    try:
        line = json.loads(line)
    except:
        continue
    f_all.append(line)
# import random
# random.shuffle(f_all)
f_all = f_all[:args.data_num]

for line in tqdm(f_write_to_read):
    line = json.loads(line)
    f_now.append(line)

# 移除键为的answer情况，然后将字典转换为其哈希值的集合
d1_set = {hash_dict(d) for d in f_all}
d2_set = {hash_dict(dict_without_key(d, ['gpt_answer', 'finish_reason'])) for d in f_now}

print("Input Num: ", len(f_all))
print("All ready ")
# 存在query重复情况
logger.info(f'实际总数据量:{len(d1_set)}')

# 计算哈希值的差集
diff_hashes = d1_set - d2_set

# 将差集的哈希值转换回字典，这就是最终要处理的数据
res = [d for d in f_all if hash_dict(d) in diff_hashes]

f_read.close()
f_write_to_read.close()
####

# 重新写入
f_write = open(write_file_name, 'a')

logger.info(f'# Data Num to be processed:{len(res)}')


def get_response_from_farui(user_prompt, model_prompt=None, system_prompt=None, image_path=None):
    try:
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': user_prompt}]
        response = dashscope.Generation.call(
            args.model_name,
            messages=messages,
            result_format='message',  # set the result to be "message" format.
            temperature=0,
            max_tokens=2000  # farui-plus的最大token数为2000,使用qwen2.5-72b时可调为3k
        )
        if response.status_code == HTTPStatus.OK:
            return response, ''
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
    except Exception as e:
        # traceback.print_exc()
        with open(args.log_file, "a") as error_log_file:
            traceback.print_exc(file=error_log_file)
        logger.info('ERROR')
        return -1, -1


# 单个进程函数
def main(data):
    lines = []
    for i, line in enumerate(tqdm(data)):
        # res, finish_reason = get_response_from_claude(user_prompt=line['user_prompt'], system_prompt=line['system_prompt'], model_prompt=line['model_prompt'])
        # res, finish_reason = get_response_from_claude(user_prompt=line['user_prompt'], system_prompt=line.get('system_prompt'), model_prompt=line.get('model_prompt'), image_path=line.get('image_path'))
        try:
            res, finish_reason = get_response_from_farui(user_prompt=line['body']['messages'][1]["content"],
                                                         system_prompt=line.get('system_prompt'))
        except:
            continue
        if res == -1:
            continue
        line['gpt_answer'] = res
        line['finish_reason'] = finish_reason
        line = json.dumps(line, ensure_ascii=False)

        # 由于中间随时可能崩掉，需要每个进程实时写入文件，但会出现乱序情况
        f_write.write(line + '\n')
        f_write.flush()

        lines.append(line)
    return lines


def multi_process(function, data, args=(), processes=4):
    chunk_size = math.ceil(len(data) / processes)
    logger.info(f'chunk_size:{chunk_size}')
    pool = Pool(processes=processes)
    results = []
    for i in range(processes):
        input_data = data[i * chunk_size:(i + 1) * chunk_size]
        res = pool.apply_async(function, args=(input_data,) + args)
        results.append(res)
    pool.close()
    pool.join()
    new_results = []
    for i in results:
        new_results.extend(i.get())
    return new_results


if __name__ == '__main__':
    data = []
    for i, line in tqdm(enumerate(res)):
        data.append(line)
        if len(data) == process_per_split * process_num:
            logger.info(i)
            multi_process(function=main, data=data, processes=process_num)
            data = []
    if len(data) > 0:
        multi_process(function=main, data=data, processes=min(process_num, len(data)))

    f_write.close()

    with open(args.log_file, "a") as error_log_file:
        error_log_file.write('ZP_FINISH')
