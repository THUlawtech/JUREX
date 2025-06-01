export DASHSCOPE_API_KEY="your_api_key"
directory="output_dictionary_after_step0" # Replace with your actual dir name
files="file_of_a_scd_category.json"  # Replace with your actual file name
scd_model="farui-plus"

python step1_batch_input_farui\&qwen.py \
    --log_file log.txt \
    --model_name ${scd_model}\
    --read_file_name "${directory}/${files}" \
    --write_file_name "${directory}/output_${scd_model}_${files}" \
    --process_per_split 1000 \
    --process_num 10 \
    --data_num 10000
