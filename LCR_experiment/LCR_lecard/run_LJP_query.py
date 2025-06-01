# @Time : 2024/7/29

# @File : run.py 
# @Project: query的罪名预测

from models import LLModel
import json
import argparse
import os

os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'  # Assign your API keys accordingly
os.environ["OPENAI_API_BASE"] = ""


def parse_args(in_args=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_type", default="qwen-plus-0806",type=str, help="model type")
    arg_parser.add_argument("--prompt", type=str,
                            default="你是一个刑法领域的律师。请根据罪名列表，对于以下案情事实，请判断它包含了罪名列表中的哪些罪名。请注意，你只需要输出罪名，不要输出多余的信息，罪名必须在罪名列表中。罪名之间用‘，’隔开。",
                            help="prompt")
    return arg_parser.parse_args(args=in_args)


if __name__ == "__main__":
    args = parse_args()

    # Load model
    if args.model_type in ['gpt-3.5-turbo', "gpt-4o-2024-08-06"]:
        model = LLModel(model=args.model_type, temperature=0,
                        api_key=os.environ.get("OPENAI_API_KEY"),
                        base_url=os.environ.get("OPENAI_API_BASE"))
    else:
        model = LLModel(model=args.model_type, temperature=0.0001, device='cuda')

    # Load data
    with open('../data/flattened_jurex4e.json') as f:
        kg_data = json.load(f)
    kg_crime_list = list(kg_data.keys())
    crimelist= f"""罪名列表：{str(kg_crime_list)}\n案情事实："""
    prompt = args.prompt + crimelist

    with open("../data/Lecardv2/test_query.json", "r") as f:
        test_data = json.load(f)

    result_data = {"query": "", "can": ""}

    with(open("../data/Lecardv2/test_query_LJP.json", "w+")) as fw:
        for i in range(len(test_data)):
            data = test_data[i]
            query = data["fact"]
            input_text = prompt + query
            raw_pred = model(input_text)
            data["pred_charge"] = raw_pred
            print(raw_pred)

            json_str = json.dumps(data,ensure_ascii=False)
            fw.write(json_str)
            fw.write("\n")
