from tqdm import tqdm
import os
import subprocess

from huggingface_hub import HfApi, logging

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
folder_names = ["bilingual", "translation", "instances"]
FOLDER_NAME = "openwebtext2"

for chunk in tqdm(range(0, 31), desc="Processing chunks"):
    chunk_string = str(chunk).zfill(2)
    CHUNK_NAME = f"ArmelR___the-pile-splitted___data___OpenWebText2___train___data-000{chunk_string}-of-00250"

    DATASET_PATH = os.path.join(FOLDER_NAME, CHUNK_NAME)

    for name in folder_names:
        filepath = str(os.path.join(DATASET_PATH, name, "data-00000-of-00001.arrow"))

        command = f"huggingface-cli upload RaiBP/openwebtext2-first-30-chunks-lang-detect-raw-output {filepath} {name}/data-000{chunk_string}-of-00030.arrow --repo-type=dataset"

        try:
            subprocess.run(command, shell=True, check=True)
            print("Command executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing the command: {e}")