from tqdm import tqdm
import subprocess
import os
from huggingface_hub import hf_hub_download
from datasets import Dataset
from transformers import GPT2TokenizerFast
from format_translation_dataset import combine_lists
from create_dataset import delete_folder

if __name__ == "__main__":
    FOLDER_NAME = "openwebtext2"
    start_id = 0
    for chunk in tqdm(range(0, 31), desc="Processing chunks"):
        chunk_string = str(chunk).zfill(2)

        file_path = hf_hub_download(repo_id="RaiBP/openwebtext2-first-30-chunks-nonenglish-ablation",
                                    repo_type="dataset",
                                    filename=f"data-000{chunk_string}-of-00030.arrow")

        original_dataset = Dataset.from_file(file_path)
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        examples = []
        small_examples = []

        for document in tqdm(original_dataset, desc="Processing documents"):
            for instance in document["instance_text"]:
                tokens = tokenizer.encode(instance, add_special_tokens=True)
                if len(tokens) == 1024:
                    examples.append({"tokens": tokens, "text": instance})
                else:
                    small_examples.append({"tokens": tokens, "length": len(tokens)})

        newline_id = tokenizer.encode("\n")[0]
        combined_examples = combine_lists(small_examples, newline_id, 1024)

        combined_examples_sorted = sorted(combined_examples, key=lambda x: x['length'])
        text_combined_examples = []
        for example in tqdm(combined_examples_sorted, desc="Decoding examples", total=len(combined_examples_sorted)):
            text_combined_examples.append(tokenizer.decode(example["tokens"]))

        text_examples = text_combined_examples + [example["text"] for example in examples]

        end_id = start_id + len(text_examples)

        english_only_dataset_formatted = Dataset.from_dict(
            {"text": text_examples, "id": list(range(start_id, end_id))})

        start_id = end_id

        try:
            os.remove(file_path)
            print(f"The file {file_path} has been deleted successfully.")
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}")

        english_only_dataset_formatted.save_to_disk(f"english_only_{chunk_string}", num_shards=1)

        os.rename(f"english_only_{chunk_string}/data-00000-of-00001.arrow",
                  f"english_only_{chunk_string}/data-000{chunk_string}-of-00030.arrow")

        command = f"huggingface-cli upload RaiBP/openwebtext2-first-30-chunks-english-only-examples english_only_{chunk_string}/data-000{chunk_string}-of-00030.arrow data/data-000{chunk_string}-of-00030.arrow --repo-type=dataset"

        try:
            subprocess.run(command, shell=True, check=True)
            print("Command executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing the command: {e}")

        delete_folder(f"english_only_{chunk_string}")