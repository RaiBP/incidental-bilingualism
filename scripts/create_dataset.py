import os
import subprocess
import shutil
import numpy as np

from datasets import Dataset
from huggingface_hub import hf_hub_download
from transformers import GPT2TokenizerFast
from tqdm import tqdm

from main import split_text_into_instances


def delete_folder(folder_path):
    try:
        # Delete the entire folder and its contents
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents successfully deleted.")
    except Exception as e:
        print(f"Error deleting folder '{folder_path}': {e}")


def ablate_document(document, exclude_dict):
    document_idx = document["document_id"]
    if document_idx in exclude_dict.keys():
        exclude_idx = exclude_dict[document_idx]
        document["instance_text"] = [text for idx, text in enumerate(document["instance_text"]) if
                                     idx not in exclude_idx]
    return document


def ablate_dataset(dataset, exclude_dict):
    return dataset.map(
        lambda document: ablate_document(document, exclude_dict),
    )


def excluded_instances_union(dict1, dict2):
    result = {}
    for key in set(dict1.keys()) | set(dict2.keys()):
        result[key] = list(set(dict1.get(key, []) + dict2.get(key, [])))
    return result


def ablate_translation(translations):
    """ Find all translation instances to remove """
    exclude_idx = {}
    for translation in tqdm(translations, desc="Finding translation instances to remove", total=len(translations)):
        document_idx = translation["document_index"]
        if document_idx not in exclude_idx.keys():
            exclude_idx[document_idx] = []
        if translation["instance_index"] not in exclude_idx[document_idx]:
            # we can have multiple translations for the same instance, but we only want to remove it once
            exclude_idx[document_idx].append(translation["instance_index"])
    return exclude_idx


def ablate_bilingual(bilinguals):
    """ Find all bilingual instances to remove """
    exclude_idx = {}
    for document_index, bilingual in tqdm(enumerate(bilinguals), desc="Finding bilingual instances to remove",
                                          total=len(bilinguals)):
        bilingual_array = np.array(bilingual['instance_labels'])
        bi_instances = np.where(bilingual_array == 'bi')[0].tolist()
        if bi_instances:
            exclude_idx[document_index] = bi_instances

    return exclude_idx


def ablate_non_english(bilinguals):
    """ Find all non-English monolingual instances """
    exclude_idx = {}
    for document_index, bilingual in tqdm(enumerate(bilinguals),
                                          desc="Finding non-English monolingual instances to remove",
                                          total=len(bilinguals)):
        if 'mono' in bilingual['instance_labels']:
            # if the most common language is not English, the instance is considered non-English
            non_english_instances = []
            for instance_index, language_tags in enumerate(bilingual['instance_tags']):
                unique_languages, counts = np.unique(language_tags, return_counts=True)
                max_count_index = np.argmax(counts)
                most_frequent_language = unique_languages[max_count_index]
                if most_frequent_language != 'en':
                    non_english_instances.append(instance_index)
            if non_english_instances:
                exclude_idx[document_index] = non_english_instances
    return exclude_idx


if __name__ == "__main__":
    FOLDER_NAME = "openwebtext2"
    tokenize = True
    for chunk in tqdm(range(16, 31), desc="Processing chunks"):
        chunk_string = str(chunk).zfill(2)

        CHUNK_NAME = f"ArmelR___the-pile-splitted___data___OpenWebText2___train___data-000{chunk_string}-of-00250"
        DATASET_PATH = os.path.join(FOLDER_NAME, CHUNK_NAME)

        bilingual_dataset = Dataset.from_file(os.path.join(DATASET_PATH, "bilingual", "data-00000-of-00001.arrow"))
        translation_dataset = Dataset.from_file(os.path.join(DATASET_PATH, "translation", "data-00000-of-00001.arrow"))

        if tokenize:
            # I tokenize the dataset again because I forgot to put the document ID in each instance in the first place when
            # calculating the bilingual instances
            file_path = hf_hub_download(repo_id="ArmelR/the-pile-splitted", repo_type="dataset",
                                        filename="data/OpenWebText2"
                                                 f"/train/data-000{chunk_string}-of"
                                                 "-00250.arrow")

            original_dataset = Dataset.from_file(file_path)
            document_ids = range(len(original_dataset))
            original_dataset = original_dataset.add_column('document_id', document_ids)
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token

            instances_dataset = original_dataset.map(
                lambda document: split_text_into_instances(document, tokenizer, 1024),
                batched=True, batch_size=1000, num_proc=1, remove_columns=original_dataset.column_names,
                desc=f"Extracting instances of 1024 tokens")

            instances_dataset.save_to_disk(os.path.join(DATASET_PATH, "instances"), num_shards=1)

        else:
            instances_dataset = Dataset.from_file(os.path.join(DATASET_PATH, "instances", "data-00000-of-00001.arrow"))

        excluded_translation_dict = ablate_translation(translation_dataset)
        excluded_bilingual_dict = ablate_bilingual(bilingual_dataset)
        excluded_non_english_dict = ablate_non_english(bilingual_dataset)

        bilingual_ablation = excluded_instances_union(excluded_bilingual_dict, excluded_translation_dict)
        non_english_ablation = excluded_instances_union(excluded_non_english_dict, bilingual_ablation)

        ablated_translation_dataset = ablate_dataset(instances_dataset, excluded_translation_dict)
        ablated_bilingual_dataset = ablate_dataset(instances_dataset, bilingual_ablation)
        ablated_non_english_dataset = ablate_dataset(instances_dataset, non_english_ablation)

        ablated_translation_dataset.save_to_disk(f"ablated_translation_{chunk_string}", num_shards=1)
        ablated_bilingual_dataset.save_to_disk(f"ablated_bilingual_{chunk_string}", num_shards=1)
        ablated_non_english_dataset.save_to_disk(f"ablated_nonenglish_{chunk_string}", num_shards=1)

        os.rename(f"ablated_translation_{chunk_string}/data-00000-of-00001.arrow",
                  f"ablated_translation_{chunk_string}/data-000{chunk_string}-of-00030.arrow")
        os.rename(f"ablated_bilingual_{chunk_string}/data-00000-of-00001.arrow",
                  f"ablated_bilingual_{chunk_string}/data-000{chunk_string}-of-00030.arrow")
        os.rename(f"ablated_nonenglish_{chunk_string}/data-00000-of-00001.arrow",
                  f"ablated_nonenglish_{chunk_string}/data-000{chunk_string}-of-00030.arrow")


        def command_string(string):
            return f"huggingface-cli upload RaiBP/openwebtext2-first-30-chunks-{string}-ablation ablated_{string}_{chunk_string}/data-000{chunk_string}-of-00030.arrow --repo-type=dataset"


        def run_command(string):
            command = command_string(string)
            # Run the command
            try:
                subprocess.run(command, shell=True, check=True)
                print("Command executed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error executing the command: {e}")


        run_command("translation")
        run_command("bilingual")
        run_command("nonenglish")

        # Call the function to delete the folder and its contents
        delete_folder(f"ablated_translation_{chunk_string}")
        delete_folder(f"ablated_nonenglish_{chunk_string}")
        delete_folder(f"ablated_bilingual_{chunk_string}")
