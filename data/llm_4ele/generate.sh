# Set environment variables for Farui model
# 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
export DASHSCOPE_API_KEY="your_api_key"

# Define the Python script path
SCRIPT_PATH="LLm_generated_charge4ele.py"

# Run the Python script with Farui model
python $SCRIPT_PATH --model_type "qwen2.5-72b-instruct" --law_type "lawdetail"

# qwen2.5-72b-instruct:
# python $SCRIPT_PATH --model_type "qwen2.5-72b-instruct" --law_type "lawdetail"

# gpt-4o:
# export `OPENAI_API_KEY` and `OPENAI_API_BASE`
# python $SCRIPT_PATH --model_type "gpt-4o-2024-08-06" --law_type "lawdetail"