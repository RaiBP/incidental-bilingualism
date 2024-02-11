from datasets import load_dataset
import json
from tqdm import tqdm

# Specify the dataset name
dataset_name = "RaiBP/openwebtext2-first-30-chunks-lang-detect-raw-output"

# Load the dataset
translation_dataset = load_dataset(dataset_name, data_dir="translation")

dataset = translation_dataset["train"]
n_examples = len(dataset)
total_instances = 0
counts_dict = {"de": 0, "fr": 0, "es": 0, "pt": 0, "it": 0, "nl": 0}
others_count = 0
instances = {}
for document in tqdm(dataset, total=n_examples):
    embedded_label = document["embedded_label"]
    primary_label = document["primary_label"]
    document_id = document["document_index"]
    instance_id = document["instance_index"]
    id = f"{document_id}-{instance_id}"
    if id not in instances.keys():
        instances[id] = [f"{embedded_label}-{primary_label}"]
    else:
        instances[id].append(f"{embedded_label}-{primary_label}")

for id, labels in instances.items():
    state = 0
    found_langs = []
    for langs in labels:
        lang_pair = langs.split("-")
        if "en" in lang_pair:
            non_english = lang_pair[0] if lang_pair[1] == "en" else lang_pair[1]
            if non_english in counts_dict.keys():
                state = 1  # found a translation with English and a language in the counts_dict
                found_langs.append(non_english)
            elif state != 1:
                state = 2  # found a translation with English and a language not in the counts_dict
        elif state != 1:
            state = 2  # found a translation without English
    if state == 1:
        majority_lang = max(set(found_langs), key=found_langs.count)
        counts_dict[majority_lang] += 1
    elif state == 2:
        others_count += 1
    else:
        print("Error: state is 0")

# Specify the file path where you want to save the JSON file
file_path = "translation_counts.json"
counts_dict["others"] = others_count

# Save the dictionary as a JSON file
with open(file_path, "w") as json_file:
    json.dump(
        counts_dict, json_file, indent=2
    )  # indent argument is optional, but it makes the file more human-readable
