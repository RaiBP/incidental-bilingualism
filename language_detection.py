from datasets import load_dataset
import glob
import subprocess
import sys
import re
from tqdm import tqdm
import concurrent.futures


def classify_instance(labels, N):
    consecutive_count = 1
    previous_label = labels[0]
    detected_languages = []
    sequences_list = []
    languages_list = []
    current_index = 0
    current_sequence = [0]

    for label in labels[1:]:
        current_index += 1
        if label == previous_label:
            consecutive_count += 1
            current_sequence.append(current_index)
            if consecutive_count > N:
                sequences_list.append(current_sequence)
                languages_list.append(label)
                if label not in detected_languages:
                    detected_languages.append(label)
                consecutive_count = 0
                current_sequence = []
        else:
            consecutive_count = 1
            current_sequence = [current_index]
        previous_label = label

    return "mono" if ((not detected_languages) or (len(detected_languages) == 1)) else "bi", detected_languages, sequences_list, languages_list


# Add the directory containing the script to sys.path
sys.path.append("")

# Define a pattern to match the desired Arrow files
file_pattern = "./Pile-CC--data-00000-of-00455--instances/data-0000*-of-00002.arrow"

# Use glob to get a list of file paths that match the pattern
file_paths = glob.glob(file_pattern)

dataset = load_dataset("arrow", data_files=file_paths)

# Specify the path to the Python script you want to run
coswid_path = "/home/rai/Documents/MSCE/internship/code_switching_detection/coswid/src/coswid.py"


def process_instance(instance):
    try:
        coswid_arguments = ["-m", "FILTER2", "-t", instance, "-c", "2", "-f", "0", "-g", "0.1", "-v", "dico"]
        process = subprocess.Popen(["python3", coswid_path] + coswid_arguments, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        # Define the regular expression pattern
        pattern = r'\[\d+\] :'

        # Split the string based on the pattern
        parts = re.split(pattern, stdout)

        # we don't need the first part
        parts = parts[1:]

        word_list = []
        label_list = []

        # Loop through the parts and print them
        for part in parts:
            # Split the part into lines and select the last line
            lines = part.strip().splitlines()
            if lines:
                last_line = lines[-1]
                try:
                    word = last_line.split(":")[0].replace(" ", "")
                    label = last_line.split(":")[1].split("(")[0].replace(" ", "")

                    word_list.append(word)
                    label_list.append(label)
                except IndexError:
                    continue

                # print(f"Word: {word}, Label: {label}")  # Use strip() to remove leading/trailing whitespace
        if label_list:
            mono_or_bi, detected_langs, sequences, sequences_langs = classify_instance(label_list, 10)

            sequences_words_list = []
            for sequence in sequences:
                sequences_words = [word_list[i] for i in sequence]
                sequences_words_list.append(sequences_words)

            if mono_or_bi == "bi":
                print(f"\nCode-switching detected: {detected_langs}\n")
                for index, sequence in enumerate(sequences_words_list):
                    print(f"{sequences_langs[index]}: {sequence}\n")

    except subprocess.CalledProcessError as e:
        print("Error running the script:", e)


# Specify the number of worker processes based on your system's capabilities
num_workers = 16

# Create a tqdm progress br
with tqdm(total=len(dataset["train"]["instances"])) as pbar:
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use tqdm to wrap the executor map for progress tracking#
        for _ in tqdm(executor.map(process_instance, dataset["train"]["instances"]),
                      total=len(dataset["train"]["instances"])):
            pbar.update(1)  # Update the progress bar
