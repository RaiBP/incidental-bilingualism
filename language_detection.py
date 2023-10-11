import subprocess
import re

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

    return "mono" if ((not detected_languages) or (
                len(detected_languages) == 1)) else "bi", detected_languages, sequences_list, languages_list


def process_instance(instance, coswid_path, coswid_model):
    try:
        coswid_arguments = ["-m", coswid_model, "-t", instance, "-c", "2", "-f", "0", "-g", "0.1", "-v", "dico"]
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

        if label_list:
            mono_or_bi, detected_langs, sequences, sequences_langs = classify_instance(label_list, 10)

            sequences_words_list = []
            for sequence in sequences:
                sequences_words = [word_list[i] for i in sequence]
                sequences_words_list.append(sequences_words)

            return {'label': mono_or_bi, 'instances': sequences_words_list, 'languages': sequences_langs}

    except subprocess.CalledProcessError as e:
        print("Error running the script:", e)
