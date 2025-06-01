import json
import argparse
import re

KEYMAP = ["客体/对象", "客观方面-行为", "客观方面-结果", "主体", "主观方面"]


def deal_batch_input(custom_id, model_type, content):
    """
    Create a batch input dictionary for API requests.
    """
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_type,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content}
            ],
            "max_tokens": 3000
        }
    }


def deal_pred(pred_charge):
    """
    Process predicted charges to ensure proper formatting.
    """
    if pred_charge and pred_charge.endswith("。"):
        pred_charge = pred_charge[:-1]
    pred = re.split(r'[，\n, （]', pred_charge)
    for i, x in enumerate(pred):
        x = x.strip()
        if x and not x.endswith("罪"):
            x += "罪"
        pred[i] = x
    return pred


def parse_args(in_args=None):
    """
    Parse command-line arguments.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_type", default="gpt-4o-2024-08-06", type=str, help="SCD Model type")
    arg_parser.add_argument("--prompt", type=str,
                            default="你是一个刑法领域的专家。请根据中国刑法知识，判断以下案情事实属于候选罪名中的哪一项。",
                            help="Prompt for the model")
    arg_parser.add_argument("--input_type", default="farui-plus", type=str, help="The input type of charge four-elements or scd methods(e.g. common)")
    return arg_parser.parse_args(args=in_args)


def load_data(file_path):
    """
    Load JSON data from a file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_ref_data(scd_data, four_ele_data):
    """
    Generate reference data for categories and crimes.
    """
    ref = {}
    for category in scd_data:
        ref[category] = ""
        for crime in scd_data[category]:
            crime += "罪"
            ref[category] += f"\n{crime}" + json.dumps(four_ele_data[crime], ensure_ascii=False, indent=4)
    return ref


def create_batch_input_files(scd_data, ref, input_type, model_type, prompt):
    """
    Create batch input files for each category.
    """
    law = {}
    with open("/Users/mac/Documents/GitHub/legalKG/data/law/刑法.json", "r", encoding='utf-8') as f:
        idx2law = json.load(f)
    with open("/Users/mac/Documents/GitHub/legalKG/data/law/罪名-法条.json", "r", encoding='utf-8') as f:
        crime2idx = json.load(f)
    for category in scd_data:
        law[category] = ""
        crimes = [crime + "罪" for crime in scd_data[category]]
        for crime in crimes:
            idx_list = crime2idx[crime[:-1]]
            article = [idx2law[str(id)] for id in idx_list]
            law[category] += f"\n{crime}：{'、'.join(article)}"


    for category in scd_data:
        output_file = f"{input_type}/gcidata_total_{category}_batch_input.json"
        with open(output_file, "w+", encoding="utf-8") as fw:
            data = scd_data[category]
            if input_type == "common":
                ref_text = {
                    "F-E": "诈骗、敲诈勒索",
                    "AP-DD": "滥用职权、玩忽职守",
                    "E-MPF": "贪污、挪用公款"
                }
                can_crime = f"""候选罪名如下： {ref_text[category]}
                据此，请判断以下案情事实属于哪个候选罪名。请注意，你只需要输出罪名，不要输出其他信息。案情事实：
                """
            elif input_type == "comlaw":
                can_crime = f"""候选罪名及相关法条如下： {law[category]}
                据此，请判断以下案情事实属于哪个候选罪名。请注意，你只需要输出罪名，不要输出其他信息。案情事实：
                """
            elif input_type == "cot":
                can_crime = f"""请从四要件理论的角度逐步进行思考：首先，从以下案情事实中提取四要件，包括：犯罪客体（指某种抽象的社会利益的具像化，比如侵犯人身利益的对象是生命权，侵犯财产利益案件的对象是手机、钱包等）、犯罪客观方面（指犯罪活动的客观事实，包括触发犯罪的关键行为，如偷、抢，和行为所导致的结果，如重伤、死亡、财产损失等）、犯罪主体（通常为一般主体，但是有特殊主体，比如贪污罪的主体是国家工作人员）、犯罪主观方面（故意或者过失）\n，然后，判断它属于候选罪名中的哪一项。候选罪名如下：{ref[category]}
                输出格式：\n【分析】：思考过程 \n【罪名】：罪名\n请注意，在【罪名】部分，你只需要输出候选罪名中的罪名，不要输出其他信息。案情事实：
                """
            else:
                can_crime = f"""候选罪名及罪名的四要件如下：{ref[category]}。\n四要件代表了罪名的四个核心要素，请对比案情事实，判断其更加符合哪一个罪名的四要件，从而判断罪名。
                请注意，你只需要输出罪名，不要输出其他信息。案情事实：
                """
            full_prompt = prompt + can_crime
            print(full_prompt)
            id_list = []
            for crime in data:
                for case in data[crime]:
                    query = case["fact"]
                    input_text = full_prompt + query
                    case_id = f"{input_type}-{crime}-{str(case['id'])}"
                    assert case_id not in id_list, f"Duplicate ID detected: {case_id}"
                    id_list.append(case_id)
                    cur_can = deal_batch_input(case_id, model_type, input_text)
                    fw.write(json.dumps(cur_can, ensure_ascii=False))
                    fw.write("\n")


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Load llm_generated four-elements: results_json_llm4ele_{args.model_type}_{args.law_type}.json
    # or use expert four-elements：data/flattened_jurex4e.json
    four_ele_data = load_data("results_json_llm4ele_{args.input_type}_{args.law_type}.json")
    scd_data = load_data("scddata_preprocessed.json")

    # Prepare reference data
    ref = generate_ref_data(scd_data, four_ele_data)

    # Create batch input files
    create_batch_input_files(scd_data, ref, args.input_type, args.model_type, args.prompt)