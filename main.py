import json
import os
import re
from functools import partial
from transformers import GPT2TokenizerFast, BertModel, BertTokenizerFast
from huggingface_hub import hf_hub_download
from datasets import Dataset
import concurrent.futures
from tqdm import tqdm
import argparse
from language_detection import detect_code_switching
from translation_mining import (
    sentence_breaker,
    extract_embedded_and_primary_sentences,
    apply_filters,
    detect_translations,
)
import warnings

# Suppressing the warning
warnings.filterwarnings(
    "ignore",
    message=".*sequence length is longer than the specified maximum sequence length.*",
)


# Define a function to split text into 1024-token instances
def split_text_into_instances(document, tokenizer, max_tokens=1024):
    tokens_batch = tokenizer.batch_encode_plus(
        document["text"], add_special_tokens=True
    )["input_ids"]
    instances_decoded = []
    for tokens in tokens_batch:
        batch_instance_tokens = []
        for i in range(0, len(tokens), max_tokens):
            instance_tokens = tokens[i : i + max_tokens]
            batch_instance_tokens.append(instance_tokens)
        instances_decoded.append(
            tokenizer.batch_decode(batch_instance_tokens, skip_special_tokens=True)
        )
    return {"instance_text": instances_decoded, "document_id": document["document_id"]}


def process_document(
    document, language_detector_path, language_detector_model, consecutive_threshold
):
    instance_label_list = []
    instance_words_list = []
    instance_tags_list = []
    instance_groups_list = []
    instance_languages_list = []
    instance_document_id = -1

    for instance in document:
        instance_results = detect_code_switching(
            instance,
            language_detector_path,
            language_detector_model,
            consecutive_threshold,
        )
        if instance_results:
            (
                instance_label,
                instance_words,
                instance_tags,
                instance_groups,
                instance_languages,
            ) = instance_results
            instance_label_list.append(instance_label)
            instance_words_list.append(instance_words)
            instance_tags_list.append(instance_tags)
            instance_groups_list.append(instance_groups)
            instance_languages_list.append(instance_languages)
            instance_document_id = document["document_id"]

    return (
        instance_label_list,
        instance_words_list,
        instance_tags_list,
        instance_groups_list,
        instance_languages_list,
        instance_document_id,
    )


def bilingual_detection(
    num_workers,
    dataset,
    language_detector_path,
    language_detector_model,
    consecutive_threshold,
):
    n_examples = len(dataset)
    document_label_list = []
    document_words_list = []
    document_tags_list = []
    document_groups_list = []
    document_languages_list = []
    document_id_list = []

    partial_process_document = partial(
        process_document,
        language_detector_path=language_detector_path,
        language_detector_model=language_detector_model,
        consecutive_threshold=consecutive_threshold,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(partial_process_document, dataset["instance_text"]),
                total=n_examples,
                desc=f"Classifying instances between monolingual and bilingual",
            )
        )

    for document_results in results:
        (
            instances_label,
            instances_words,
            instances_tags,
            instances_groups,
            instances_languages,
            instance_document_id,
        ) = document_results
        document_label_list.append(instances_label)
        document_words_list.append(instances_words)
        document_tags_list.append(instances_tags)
        document_groups_list.append(instances_groups)
        document_languages_list.append(instances_languages)
        document_id_list.append(instance_document_id)

    results_dict = {
        "instance_labels": document_label_list,
        "instance_words": document_words_list,
        "instance_tags": document_tags_list,
        "instance_groups": document_groups_list,
        "instance_languages": document_languages_list,
        "instance_document_id": document_id_list,
    }
    return Dataset.from_dict(results_dict)


def translation_detection(dataset):
    translation_tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
    translation_model = BertModel.from_pretrained("setu4993/LaBSE")
    translation_model = translation_model.eval()

    embedded_label_list = []
    primary_label_list = []
    embedded_sentence_list = []
    primary_sentence_list = []
    instance_index_list = []
    document_index_list = []

    for document_index, document in tqdm(
        enumerate(dataset), desc="Finding translation pairs", total=len(dataset)
    ):
        for instance_index, instance_label in enumerate(document["instance_labels"]):
            if instance_label == "bi":
                sentences, sentence_labels = sentence_breaker(
                    document["instance_words"][instance_index],
                    document["instance_tags"][instance_index],
                )
                if len(set(sentence_labels)) > 1:
                    (
                        embedded_sentences,
                        primary_sentences,
                        embedded_label,
                        primary_label,
                    ) = extract_embedded_and_primary_sentences(
                        sentences, sentence_labels
                    )
                    translation_pairs = detect_translations(
                        embedded_sentences,
                        primary_sentences,
                        translation_tokenizer,
                        translation_model,
                    )
                    for sentence_embedded, sentence_primary in translation_pairs:
                        if apply_filters(sentence_embedded, sentence_primary):
                            embedded_label_list.append(embedded_label)
                            primary_label_list.append(primary_label)
                            embedded_sentence_list.append(sentence_embedded)
                            primary_sentence_list.append(sentence_primary)
                            instance_index_list.append(instance_index)
                            document_index_list.append(document_index)

    results_dict = {
        "embedded_label": embedded_label_list,
        "primary_label": primary_label_list,
        "embedded_sentence": embedded_sentence_list,
        "primary_sentence": primary_sentence_list,
        "instance_index": instance_index_list,
        "document_index": document_index_list,
    }
    return Dataset.from_dict(results_dict)


def count_bilingual_instances(dataset):
    total_instances = 0
    bilingual_instances = 0
    for document in tqdm(dataset, desc="Counting bilingual instances"):
        total_instances += len(document["instance_labels"])
        bilingual_instances += document["instance_labels"].count("bi")
    return bilingual_instances, total_instances


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str)
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--coswid_model", type=str, default="FILTER2")
    parser.add_argument("--coswid_path", type=str, default="./coswid/src/coswid.py")
    parser.add_argument("--cache_dir", type=str)

    return parser.parse_args()


def main():
    # Load the GPT-2 tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    args = parse_args()

    coswid_path = args.coswid_path
    coswid_model = args.coswid_model

    pattern = r"\.\w+"
    output_filename = (
        re.sub(pattern, "", args.filename).replace("./", "").replace("/", "___")
    )

    if args.repo_id:
        if args.cache_dir:
            file_path = hf_hub_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                filename=args.filename,
                cache_dir=args.cache_dir,
            )
        else:
            file_path = hf_hub_download(
                repo_id=args.repo_id, repo_type="dataset", filename=args.filename
            )
        output_filename = args.repo_id.replace("/", "___") + "___" + output_filename
    else:
        file_path = args.filename

    results_folder = "./" + output_filename
    if not os.path.exists(results_folder):
        # Create the directory if it does not exist
        os.makedirs(results_folder)

    dataset = Dataset.from_file(file_path)
    document_ids = range(len(dataset))
    dataset = dataset.add_column("document_id", document_ids)

    if "instance_text" not in dataset.column_names:
        instances_dataset = dataset.map(
            lambda document: split_text_into_instances(
                document, tokenizer, args.max_tokens
            ),
            batched=True,
            batch_size=1000,
            num_proc=1,
            remove_columns=dataset.column_names,
            desc=f"Extracting instances of {args.max_tokens} tokens",
        )
        instances_dataset.save_to_disk(results_folder + "/instances", num_shards=1)
        dataset = instances_dataset

        print(
            "Finished extracting instances. Instances dataset saved at "
            + results_folder
            + "/instances"
        )
    else:
        print("Instances column found. Skipping instance extraction.")

    if "instance_labels" not in dataset.column_names:
        bilingual_dataset = bilingual_detection(
            args.num_workers, dataset, coswid_path, coswid_model, args.N
        )
        bilingual_dataset.save_to_disk(results_folder + "/bilingual", num_shards=1)
        dataset = bilingual_dataset
        print(
            "Finished classifying instances. Instance classification dataset saved at "
            + results_folder
            + "/bilingual"
        )
    else:
        print(
            "Bilingual classification labels column found. Skipping instance classification."
        )

    num_bilingual_instances, num_total_instances = count_bilingual_instances(dataset)
    percentage_bilingual = num_bilingual_instances / num_total_instances * 100
    print("Counting bilingual instances...")
    print(
        f"Found {num_bilingual_instances} bilingual instances out of {num_total_instances} total instances "
        f"({percentage_bilingual:.2f}%)."
    )

    translation_dataset = translation_detection(dataset)
    translation_dataset.save_to_disk(results_folder + "/translation", num_shards=1)
    num_translation_instances = len(set(translation_dataset["instance_index"]))
    percentage_translation = num_translation_instances / num_total_instances * 100

    print(
        "Finished finding translation pairs. Translation pairs dataset saved at "
        + results_folder
        + "/translation"
    )

    print(
        f"Found {num_translation_instances} translation instances out of {num_total_instances} total instances "
        f"({percentage_translation:.2f}%)."
    )


if __name__ == "__main__":
    main()
