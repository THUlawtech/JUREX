import json
import argparse
import os
from models import LLModel

KEYMAP = ["客体/对象", "客观方面-行为", "客观方面-结果", "主体", "主观方面"]


def parse_args(in_args=None):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_type", default="gpt-4o-2024-08-06", type=str, help="model type")
    arg_parser.add_argument("--prompt", type=str,
        default="你是一个刑法领域的专家。请对给出的罪名，根据中国刑法知识，按顺序输出犯罪的四要件，包括："
                "犯罪客体（指某种抽象的社会利益的具像化，比如侵犯人身利益的对象是生命权，侵犯财产利益案件的对象是手机、钱包等）、"
                "犯罪客观方面（指犯罪活动的客观事实，包括触发犯罪的关键行为，如偷、抢，和行为所导致的结果，如重伤、死亡、财产损失等）、"
                "犯罪主体（通常为一般主体，但是有特殊主体，比如国家工作人员）、"
                "犯罪主观方面（故意或者过失）",
        help="prompt")
    arg_parser.add_argument("--law_type", choices=["nolaw", "lawid", "lawdetail"], default="lawdetail", type=str)
    return arg_parser.parse_args(args=in_args)


def main():
    args = parse_args()

    # load model
    if args.model_type in ['gpt-3.5-turbo', "gpt-4o-2024-08-06"]:
        model = LLModel(model=args.model_type, temperature=0,
                        api_key=os.environ.get("OPENAI_API_KEY"),
                        base_url=os.environ.get("OPENAI_API_BASE"))
    else:
        model = LLModel(model=args.model_type, temperature=0.0001, device='cuda')

    # SCD charges.
    test_crime = ['故意伤害罪', '故意杀人罪', '过失致人死亡罪', '抢劫罪', '绑架罪',
                  '抢夺罪', '诈骗罪', '敲诈勒索罪', '滥用职权罪', '玩忽职守罪', '贪污罪', '挪用公款罪']

    # load corresponding article
    if args.law_type != "nolaw":
        with open("../law/criminal_law.json", "r", encoding='utf-8') as f:
            idx2law = json.load(f)
        with open("../law/crime2article.json", "r", encoding='utf-8') as f:
            crime2idx = json.load(f)

    results = []
    for idx, crime in enumerate(test_crime):
        ref = ""
        if args.law_type == "lawid":
            law_id = crime2idx[crime[:-1]]
            ref += f"\n相关法条：刑法第{law_id}条"
        elif args.law_type == "lawdetail":
            idx_list = crime2idx[crime[:-1]]
            law_detail = [idx2law[str(i)] for i in idx_list]
            ref += f"\n相关法条：{'、'.join(law_detail)}"

        output_format = {
            "罪名": "",
            "罪名四要件": {
                "犯罪客体": "",
                "犯罪客观方面": "",
                "犯罪主体": "",
                "犯罪主观方面": ""
            }
        }

        # fewshot = ""
        output_format = f"\n请综合以上信息，生成更能体现罪名特征的四要件。注意：请进行思考，尽可能完善的回答。输出格式：{json.dumps(output_format, ensure_ascii=False)}\n罪名："

        input_text = args.prompt + ref + output_format + crime
        # generate response
        response = model(input_text)
        results.append({
            "user_prompt": input_text,
            "crime": crime,
            "response": response
        })
    with open(f"results_json_llm4ele_{args.model_type}_{args.law_type}.json", "w", encoding="utf-8") as outfile:
        for line in results:
            json.dump(line, outfile, ensure_ascii=False)
            outfile.write("\n")


if __name__ == "__main__":
    main()
