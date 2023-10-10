from huggingface_hub import hf_hub_download
from datasets import Dataset
from transformers import GPT2TokenizerFast
import concurrent.futures
from tqdm import tqdm

file_path = hf_hub_download(repo_id="ArmelR/the-pile-splitted", repo_type="dataset",
                            filename="data/Pile-CC/train/data-00000-of-00455.arrow", cache_dir=".")

ds = Dataset.from_file(file_path)

# Load the GPT-2 tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# Define a function to split text into 1024-token instances
def split_text_into_instances(text, max_tokens=1024):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    instances = []

    for i in range(0, len(tokens), max_tokens):
        instance_tokens = tokens[i:i + max_tokens]

        instance_text = tokenizer.decode(instance_tokens, skip_special_tokens=True)
        instances.append(instance_text)

    return instances


# Process the dataset using parallel processing
instances = []


def process_document(document):
    document_instances = split_text_into_instances(document)
    return document_instances


# Specify the number of worker processes based on your system's capabilities
num_workers = 16

# Create a tqdm progress bar
with tqdm(total=len(ds["text"])) as pbar:
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use tqdm to wrap the executor map for progress tracking#
        for document_instance_list in tqdm(executor.map(process_document, ds["text"]), total=len(ds["text"])):
            instances.extend(document_instance_list)
            pbar.update(1)  # Update the progress bar
