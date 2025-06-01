# @Time : 2024/9/5 14:23 

# @File : eval_sample.py 
# @Project: PyCharm
import json
import csv
import os
from sklearn import metrics


def find_dict_by_key_value(dict_list, key, value):
    for dictionary in dict_list:
        if key in dictionary and dictionary[key] == value:
            return dictionary
    return None


def deal_label(accusation, prediction, i):
    type = ref[i]
    try:
        pred = type.index(prediction)
    except:
        pred = -1
    if prediction in accusation and prediction in type:  # 处理可能有两个标签的情况
        label = type.index(prediction)
        return label, pred
    for case in accusation:
        if case in type:
            label = type.index(case)
            break
    return label, pred


ref1 = [
    "F-E", "AP-DD", "E-MPF"]

ref = {
    2: ['诈骗', '敲诈勒索'],
    3: ['滥用职权', '玩忽职守'],
    4: ['贪污', '挪用公款']
}

input_types = [ ""]
model_type = ""


sample_data = []
with open("/Users/mac/Documents/GitHub/legalKG/data/GCI/data.json", "r") as f:
    for line in f:
        sample_data.append(json.loads(line))

for input_type in input_types:
    for name in [2,3,4]:
        write_case=[]
        result = []
        labels_all, preds_all = [], []
        correct_predictions = 0
        print(ref[name])
        count = 0
        fdir = f"{input_type}/output_{model_type}_gcidata_total_{ref1[name]}_batch_input.json"
        with open(fdir, "r", encoding='utf-8') as file:
            for line in file:
                # Parsing the JSON string into a dict and appending to the list of results
                res = json.loads(line.strip())
                case = {"id": res['custom_id']}
                case['pred'] = res['gpt_answer']    # if use gpt-4o: ['output']['choices'][0]['message']['content']
                if input_type == "cot":
                    case['pred'] = case['pred'].split("【罪名】：")[-1]
                    for crime in ref[name]:
                        if crime in case['pred']:
                            case['pred'] = crime
                            break
                if "罪" in case['pred']:
                    case['pred'] = case['pred'].split("罪")[0]

                case_ori = sample_data[int(res['custom_id'].split("-")[-1])]
                case['accusation'] = case_ori['accusation']

                case['fact'] = case_ori['fact']
                # if len(case_ori["fact"])<60:
                #     continue
                if case['pred'] in case['accusation']:  # 去掉“罪”
                    correct_predictions += 1
                else:
                    write_case.append(case)

                label, pred = deal_label(case['accusation'], case['pred'], name)
                if pred == -1:
                    count += 1
                labels_all.append(label)
                preds_all.append(pred)

                result.append(case)

        report = metrics.classification_report(labels_all, preds_all, target_names=ref[name].append("-1"), digits=4)
        print(report)
        total_cases = len(result)
        # Calculate accuracy
        accuracy = correct_predictions / total_cases

        # Print the accuracy
        print(f"Prediction Accuracy: {accuracy:.2%}")
        print(count)


