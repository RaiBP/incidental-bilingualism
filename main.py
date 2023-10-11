from transformers import GPT2TokenizerFast
from huggingface_hub import hf_hub_download
from datasets import Dataset
import concurrent.futures
from tqdm import tqdm
import argparse
from language_detection import process_instance
from concurrent.futures import ProcessPoolExecutor, as_completed

COSWID_PATH = None  # Define the variable at the module level
COSWID_MODEL = None  # Define the variable at the module level

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
            if bilingual_detection_dict['label'] == 'bi':
                print(bilingual_detection_dict)

    return label_list


def process_document(document):
    document_label = split_text_into_instances(document)
    return document_label


def get_instances(num_workers, ds, n_examples):
    # Create a tqdm progress bar
    with tqdm(total=n_examples) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Use tqdm to wrap the executor map for progress tracking#
            for _ in tqdm(executor.map(process_document, ds["text"]), total=n_examples):
                pbar.update(1)  # Update the progress bar


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

    file_path = hf_hub_download(repo_id=args.repo_id, repo_type="dataset",
                                filename=args.filename, cache_dir=".")

    dataset = Dataset.from_file(file_path)
    n_examples = len(dataset)
    get_instances(args.num_workers, dataset, n_examples)

    COSWID_PATH = args.coswid_path
    COSWID_MODEL = args.coswid_model


if __name__ == "__main__":
    main()
