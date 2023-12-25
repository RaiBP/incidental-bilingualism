import os

from datasets import Dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import concurrent.futures
from functools import partial

from create_dataset import ablate_translation, ablate_bilingual, ablate_non_english
from format_translation_dataset import format_dataset, find_equal_elements_indices, remove_equal_elements, combine_lists


def set_difference(dict1, dict2):
    result = {}
    for key in set(dict1.keys()):
        if key not in dict2:
            result[key] = dict1[key]
        else:
            result[key] = list(set(dict1[key]) - set(dict2.get(key, [])))
    return result


def get_dataset(examples, small_examples, merger_token):
    equal_indices_small = find_equal_elements_indices(small_examples, key="tokens")
    equal_indices = find_equal_elements_indices(examples, key="tokens")

    filtered_examples_list = remove_equal_elements(examples, equal_indices)
    filtered_examples_small_list = remove_equal_elements(small_examples, equal_indices_small)

    combined_instances = combine_lists(filtered_examples_small_list, merger_token, 1024)

    combined_instances_sorted = sorted(combined_instances, key=lambda x: x['length'])

    combined_instances_sorted.extend(filtered_examples_list)

    text_instances = [tokenizer.decode(instance["tokens"]) for instance in combined_instances_sorted]

    return Dataset.from_dict(
        {"text": text_instances, "id": list(range(len(text_instances)))})


def process_chunk(chunk, folder_name, tokenizer, dict_type):
    chunk_string = str(chunk).zfill(2)

    filename = f"ArmelR___the-pile-splitted___data___OpenWebText2___train___data-000{chunk_string}-of-00250"
    dataset_path = os.path.join(folder_name, filename)

    translation_dataset = Dataset.from_file(os.path.join(dataset_path, "translation", "data-00000-of-00001.arrow"))
    bilingual_dataset = Dataset.from_file(os.path.join(dataset_path, "bilingual", "data-00000-of-00001.arrow"))
    instances_dataset = Dataset.from_file(os.path.join(dataset_path, "instances", "data-00000-of-00001.arrow"))

    translation_dict = ablate_translation(translation_dataset)
    bilingual_dict = ablate_bilingual(bilingual_dataset)
    if dict_type == "non-english":
        non_english_dict = ablate_non_english(bilingual_dataset)
        filtered_dict = set_difference(set_difference(non_english_dict, bilingual_dict), translation_dict)
    else:
        filtered_dict = set_difference(bilingual_dict, translation_dict)

    examples, small_examples = format_dataset(instances_dataset, filtered_dict, tokenizer)

    return {"examples": examples, "small_examples": small_examples}


if __name__ == "__main__":
    FOLDER_NAME = "openwebtext2"

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    DATASET = "non-english"

    # Create a ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        # Map the function to the range of chunks for parallel processing
        results = list(
            tqdm(executor.map(partial(process_chunk, folder_name=FOLDER_NAME, tokenizer=tokenizer, dict_type=DATASET), range(0, 31)), total=31,
                 desc="Processing chunks"))

    newline_id = tokenizer.encode("\n")[0]

    examples_list = []
    small_examples_list = []
    print("Made it here")
    for result in results:
        examples_list.extend(result["examples"])
        small_examples_list.extend(result["small_examples"])
    print("Here too")
    dataset_formatted = get_dataset(examples_list, small_examples_list, newline_id)
    dataset_formatted.push_to_hub("RaiBP/openwebtext2-first-30-chunks-nonenglish-examples")
