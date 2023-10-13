import json
import pandas as pd
from transformers import GPT2TokenizerFast
from huggingface_hub import hf_hub_download
from datasets import Dataset
import concurrent.futures
from tqdm import tqdm
import argparse
from language_detection import process_instance

COSWID_PATH = None
COSWID_MODEL = None

# Load the GPT-2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# Define a function to split text into 1024-token instances
def split_text_into_instances(text, max_tokens=1024):
    global tokenizer, COSWID_MODEL, COSWID_PATH

    tokens = tokenizer.encode(text, add_special_tokens=True)
    label_list = []

    for i in range(0, len(tokens), max_tokens):
        instance_tokens = tokens[i:i + max_tokens]

        instance_text = tokenizer.decode(instance_tokens, skip_special_tokens=True)

        bilingual_detection_dict = process_instance(instance_text, COSWID_PATH, COSWID_MODEL)
        if bilingual_detection_dict is not None:
            label_list.append(bilingual_detection_dict)

    return label_list


def process_document(document):
    document_label = split_text_into_instances(document)
    return document_label


def get_instances(num_workers, ds, n_examples):
    bilingual_data = []
    translation_data = []
    bilingual_results = {}
    translation_results = {}
    document_index = 0
    # Create a tqdm progress bar
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for document_results in tqdm(executor.map(process_document, ds["text"]), total=n_examples):
            for instance_index, instance_dict in enumerate(document_results):
                if instance_dict["label"] == "bi":
                    for index, group in enumerate(instance_dict["groups"]):
                        bilingual_dict = {"text": " ".join(group), "label": instance_dict["languages"][index],
                                          'instance_idx': instance_index, 'document_idx': document_index}
                        bilingual_data.append(bilingual_dict)
                    if instance_dict["translation_pairs"]:
                        for translation_pair in instance_dict["translation_pairs"]:
                            translation_languages = [translation_pair["embedded_label"],
                                                     translation_pair["primary_label"]]
                            translation_languages.sort()
                            translation_languages_string = "-".join(translation_languages)
                            if translation_languages_string in translation_results.keys():
                                translation_results[translation_languages_string] += 1
                            else:
                                translation_results[translation_languages_string] = 1
                            translation_pair["instance_idx"] = instance_index
                            translation_pair["document_idx"] = document_index
                            translation_data.append(translation_pair)
                instance_languages = list(set(instance_dict["languages"]))
                instance_languages.sort()
                languages_string = "-".join(instance_languages)
                if languages_string in bilingual_results.keys():
                    bilingual_results[languages_string] += 1  # increment the number of instances for this language pair
                else:
                    bilingual_results[languages_string] = 1  # initialize the number of instances for this language pair
            document_index += 1
            if document_index == 200:
                break
    return pd.DataFrame(bilingual_data), pd.DataFrame(translation_data), bilingual_results, translation_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--coswid_model", type=str, default="FILTER2")
    parser.add_argument("--coswid_path", type=str, default="./coswid/src/coswid.py")

    return parser.parse_args()


def main():
    global COSWID_MODEL, COSWID_PATH

    args = parse_args()

    COSWID_PATH = args.coswid_path
    COSWID_MODEL = args.coswid_model

    file_path = hf_hub_download(repo_id=args.repo_id, repo_type="dataset",
                                filename=args.filename, cache_dir=".")

    dataset = Dataset.from_file(file_path)
    n_examples = len(dataset)
    bilingual_df, translation_df, bilingual_results, translation_results = get_instances(args.num_workers, dataset,
                                                                                         n_examples)

    output_filename = args.repo_id.replace("/", "___") + "___" + args.filename.split(".")[0].replace("/", "___") + "___"
    bilingual_df.to_csv(output_filename + "bilingual_results.csv", index=False)
    translation_df.to_csv(output_filename + "translation_results.csv", index=False)

    with open(output_filename + "bilingual_counts.json", 'w') as f:
        json.dump(bilingual_results, f)

    with open(output_filename + "translation_counts.json", 'w') as f:
        json.dump(translation_results, f)


if __name__ == "__main__":
    main()
