import json
import tiktoken


# dataset: Liu, X., Yin, D., Feng, Y., Wu, Y., & Zhao, D. (2021). Everything has a cause: Leveraging causal inference in legal text analysis. arXiv preprint arXiv:2104.09420.
DATA_FILE = 'data.json'
OUTPUT_FILE = 'scddata_preprocessed.json'
ENCODING_MODEL = "gpt-4o"

# SCD categories and accusations
CATEGORIES = {
    "F-E": {'诈骗': [], '敲诈勒索': []},
    "AP-DD": {'滥用职权': [], '玩忽职守': []},
    "E-MPF": {'贪污': [], '挪用公款': []}
}


def num_tokens(text):
    """
    Calculate the number of tokens in a given text using the specified encoding model.
    """
    encoding = tiktoken.encoding_for_model(ENCODING_MODEL)
    return len(encoding.encode(text))


def find_category_and_accusation(accusation):
    """
    Find the category and accusation from the predefined categories.
    """
    for category, accusations in CATEGORIES.items():
        if accusation in accusations:
            return category, accusation
    return None, None


def load_json_data(file_path):
    """
    Load JSON data from a file and assign unique IDs to each element.
    """
    json_data = []
    json_data_id = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                cur = json.loads(line)
                cur['id'] = json_data_id
                json_data_id += 1
                json_data.append(cur)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {file_path}.")
    return json_data


def process_data(json_data):
    """
    Process the JSON data to categorize accusations and count occurrences.
    """
    accusation_counts = {category: {accu: 0 for accu in accusations} for category, accusations in CATEGORIES.items()}
    data_count = 0

    for element in json_data:
        accusation_list = element.get("accusation", [])
        for accusation in accusation_list:
            category, accu = find_category_and_accusation(accusation)
            if category and accu:
                CATEGORIES[category][accu].append(element)
                accusation_counts[category][accu] += 1
                data_count += num_tokens(element['fact'])

    return accusation_counts, data_count


def save_to_json(data, file_path):
    """
    Save data to a JSON file with proper formatting.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error: Failed to save data to {file_path}. {e}")


if __name__ == "__main__":
    # Load data
    json_data = load_json_data(DATA_FILE)

    # Process data
    accusation_counts, total_tokens = process_data(json_data)

    # Print accusation counts
    for category, accusations in accusation_counts.items():
        for accusation, count in accusations.items():
            print(f"{category} - {accusation}: {count}")

    print(f"Total accusations: {sum(sum(accusations.values()) for accusations in accusation_counts.values())}")
    print(f"Total tokens: {total_tokens}")

    # Save processed data
    save_to_json(CATEGORIES, OUTPUT_FILE)