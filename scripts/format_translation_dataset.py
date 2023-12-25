import os

from datasets import Dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

from create_dataset import ablate_translation


def combine_lists(lst, merger_token, max_length):
    sorted_list = sorted(lst, key=lambda x: x['length'])
    combined_tokens = []
    combined_length = 0
    result = []
    for instance in sorted_list:
        if combined_length + instance['length'] < max_length:
            combined_tokens.append(merger_token)
            combined_tokens.extend(instance['tokens'])
            combined_length = len(combined_tokens)
        else:
            result.append({'tokens': combined_tokens, 'length': combined_length})
            combined_tokens = instance['tokens']
            combined_length = instance['length']
    if combined_length:
        result.append({'tokens': combined_tokens, 'length': combined_length})
    return result


def format_dataset(instances, index_dict, tokenizer_, max_length=1024):
    result = []
    small_instances = []

    for document_idx in index_dict.keys():
        for instance_idx in index_dict[document_idx]:
            instance = instances[document_idx]["instance_text"][instance_idx]
            tokens = tokenizer_.encode(instance, add_special_tokens=True)
            if len(tokens) == max_length:
                result.append({"tokens": tokens, "text": instance})
            else:
                small_instances.append({"tokens": tokens, "length": len(tokens)})
    return result, small_instances


def find_equal_elements_indices(input_list, key="tokens"):
    seen = {}
    equal_idx = {}

    for idx, elem in enumerate(input_list):
        tokens = tuple(elem.get(key, []))
        if tokens in seen:
            equal_idx[tokens] = [seen[tokens], idx]
        else:
            seen[tokens] = idx

    return equal_idx


def remove_equal_elements(input_list, equal_idx):
    if equal_idx:
        indices_to_remove, _ = zip(*equal_idx.values())
        filtered_examples = [value for index, value in enumerate(input_list) if index not in indices_to_remove]
    else:
        filtered_examples = input_list.copy()
    return filtered_examples


if __name__ == "__main__":
    FOLDER_NAME = "openwebtext2"
    examples_list = []
    small_examples_list = []

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    small_examples_found = 0
    translation_instances_found = 0
    for chunk in tqdm(range(0, 31), desc="Processing chunks"):
        chunk_string = str(chunk).zfill(2)

        CHUNK_NAME = f"ArmelR___the-pile-splitted___data___OpenWebText2___train___data-000{chunk_string}-of-00250"
        DATASET_PATH = os.path.join(FOLDER_NAME, CHUNK_NAME)

        translation_dataset = Dataset.from_file(os.path.join(DATASET_PATH, "translation", "data-00000-of-00001.arrow"))
        instances_dataset = Dataset.from_file(os.path.join(DATASET_PATH, "instances", "data-00000-of-00001.arrow"))

        translation_dict = ablate_translation(translation_dataset)

        examples, small_examples = format_dataset(instances_dataset, translation_dict, tokenizer)

        examples_list.extend(examples)
        small_examples_list.extend(small_examples)

    equal_indices_small = find_equal_elements_indices(small_examples_list, key="tokens")
    equal_indices = find_equal_elements_indices(examples_list, key="tokens")

    filtered_examples_list = remove_equal_elements(examples_list, equal_indices)
    filtered_examples_small_list = remove_equal_elements(small_examples_list, equal_indices_small)

    newline_id = tokenizer.encode("\n")[0]
    combined_instances = combine_lists(filtered_examples_small_list, newline_id, 1024)

    combined_instances = sorted(combined_instances, key=lambda x: x['length'])
    combined_instances.extend(filtered_examples_list)
    text_instances = [tokenizer.decode(instance["tokens"]) for instance in combined_instances]

    translation_dataset_formatted = Dataset.from_dict({"text": text_instances, "id": list(range(len(text_instances)))})
