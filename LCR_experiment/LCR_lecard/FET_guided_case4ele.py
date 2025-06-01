# @Time : 2024/7/29

# @File : run.py
# @Project: PyCharm
from models import LLModel
import json
import argparse
import re
import tiktoken
import os
keymap = ["客体/对象", "客观方面-行为", "客观方面-结果", "主体", "主观方面"]

os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'  # Assign your API keys accordingly
os.environ["OPENAI_API_BASE"] = ""


def parse_args(in_args=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_type", default="gpt-4o-2024-08-06", type=str, help="model type")
    arg_parser.add_argument("--prompt", type=str,
                            default="你是一个刑法领域的专家。请对以下案情事实，根据中国刑法知识，按顺序输出："
                                    "1、犯罪的四要件，包括："
                                    "犯罪客体（指某种抽象的社会利益的具像化，比如侵犯人身利益的对象是生命权，侵犯财产利益案件的对象是手机、钱包等）、"
                                    "犯罪客观方面（指犯罪活动的客观事实，包括触发犯罪的关键行为，如偷、抢，和行为所导致的结果，如重伤、死亡、财产损失等）、"
                                    "犯罪主体（通常为一般主体，但是有特殊主体，比如国家工作人员）、"
                                    "犯罪主观方面（故意或者过失）"
                                    "2.罪名：只输出具体的罪名"
                                    "3.相关法条：只输出法条编号。",
                            help="prompt")
    return arg_parser.parse_args(args=in_args)


def deal_pred(pred_charge):
    if pred_charge.endswith("。"):
        pred_charge = pred_charge[:-1]
    pred = re.split(r'[，\n, （]', pred_charge)
    return [x.strip() + ("罪" if x.strip() and not x.strip().endswith("罪") else "") for x in pred if x.strip()]


def deal_crime(crimes, kg_crime_list):
    mapped = {
        "伪造、变造居民身份证罪": "伪造、变造、买卖身份证件罪",
        "生产、销售、提供假药罪": "生产、销售假药罪",
        "非法猎捕、杀害珍贵、濒危野生动物罪": "危害珍贵、濒危野生动物罪",
        "非法收购、运输、出售珍贵、濒危野生动物、珍贵、濒危野生动物制品罪": "危害珍贵、濒危野生动物罪",
        "非法采伐、毁坏国家重点保护植物罪": "危害国家重点保护植物罪"
    }
    clear_crime, exclude = [], []
    for c in crimes:
        crime = mapped.get(c, c)
        (clear_crime if crime in kg_crime_list else exclude).append(crime)
    return list(set(clear_crime)), list(set(exclude))


def extract_related_ids(id_to_search, file_path):
    related_ids = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            parts = line.split("\t")
            parts = [int(x) for x in parts]
            if parts[0] == id_to_search:
                related_ids.append([parts[2], parts[3]])
    return related_ids


def format_dict(data):
    return "".join([f"\n- {k}:{v} " for k, v in data.items()])


def main():
    args = parse_args()

    # Load model
    if args.model_type in ['gpt-3.5-turbo', "gpt-4o-2024-08-06"]:
        model = LLModel(model=args.model_type, temperature=0,
                        api_key=os.environ.get("OPENAI_API_KEY"),
                        base_url=os.environ.get("OPENAI_API_BASE"))
    else:
        model = LLModel(model=args.model_type, temperature=0.0001, device='cuda')

    fewshot_example = """输出格式为json。对于案件中涉及的每个罪名，分别输出为一个字典。参考样例如下：
          {"罪名1":{"犯罪的四要件":{
                "犯罪客体":"人身利益：被害人王某某的生命权、财产利益：车辆",
                "犯罪客观方面":"被告人吴某某酒后驾驶小型轿车与被害人王某某相撞，造成被害人王某某当场死亡及车辆损坏",
                "犯罪主体":"司机被告人吴某某",
                "犯罪主观方面":"过失"},
                "罪名":"交通肇事罪",
                "相关法条":"第133条"},
            "罪名2":{"犯罪的四要件":{
                "犯罪客体":"社会管理秩序...",
                "犯罪客观方面":"...",
                "犯罪主体":"...",
                "犯罪主观方面":"故意"},
                "罪名":"伪造、变造、买卖国家机关公文、证件、印章罪",
                "相关法条":"第280条第1款"}}"""
    prompt = args.prompt + fewshot_example + "\n请根据以下案情事实，输出分析结果。请注意，不要直接回答样例的内容。案情事实："

    # Load data
    with open('../data/Lecardv2/test_query_LJP.json') as f:
        sample_data = json.load(f)
    with open('../data/flattened_jurex4e.json') as f:
        kg_data = json.load(f)
    kg_crime_list = list(kg_data.keys())

    # Run model on queries
    with(open("../data/Lecardv2/test_query_LJP_simcrime_4ele.json", "w+")) as fw:
        for data in sample_data:
            pred_crime = deal_pred(data['pred_charge']) + data.get('expand_crime', [])
            pred_crime, _ = deal_crime(pred_crime, kg_crime_list)

            ref = ""
            if pred_crime:
                ref += "\n相关罪名和专家四要件（仅为参考的关键词，你需要根据案情具体分析）："
                ref += ''.join([f"{c}{format_dict(kg_data[c])}\n" for c in pred_crime])

            input_text = prompt + data["fact"] + ref + "\n你只需要输出json，不要输出多余的信息。"
            raw_pred = model(input_text)
            data["query_4element"] = raw_pred
            print(raw_pred)
            fw.write(json.dumps(data, ensure_ascii=False) + "\n")

    # Process candidate data
    can_path = '../data/Lecardv2/GEAR_candidate/candidate_gpt_GEAR/'
    can_data = [json.load(open(os.path.join(can_path, f))) for f in os.listdir(can_path)]
    for file in os.listdir(can_path):
        with open(can_path+file, "r") as f:
            can_data.append(json.load(f))

    for can in can_data:
        file_path = f"./root/bge/Lecard/GEAR_candidate/candidate_gpt_GEAR_llm_qwen25-72b/{can['pid']}.json"
        if os.path.exists(file_path):
            continue

        can_charge, exclude_can = deal_crime(can["charge"], kg_crime_list)
        ref = ""
        if can_charge:
            ref += f"\n相关罪名: {can_charge}. 和专家四要件（仅为参考的关键词，你需要根据案情具体分析）："
            ref += ''.join([f"{c}{format_dict(kg_data[c])}\n" for c in can_charge])
        if exclude_can:
            ref += f"\n其他参考罪名：{exclude_can}"    # 可能candidate有些罪名不在标注罪名里

        input_text = prompt + can["fact"] + ref + "\n你只需要输出json，不要输出多余的信息。"
        raw_pred = model(input_text)
        can["can_4element"] = raw_pred

        with open(file_path, 'w') as fw:
                json.dump(can, fw, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
