import subprocess
import re


def classify_instance(labels, N):
    groups = []
    current_group = []

    for i, label in enumerate(labels):
        if i == 0 or label == labels[i - 1]:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]

    groups.append(current_group)

    valid_groups_indices = [index for index, group in enumerate(groups) if len(group) > N]

    valid_groups = [groups[i] for i in valid_groups_indices]
    valid_groups_labels = [labels[group[0]] for group in valid_groups]

    return "bi" if len(set(valid_groups_labels)) > 1 else "mono", valid_groups, groups


def classify_instance_legacy(labels, N):
    consecutive_count = 1
    previous_label = labels[0]
    detected_languages = []
    sequences_list = []
    languages_list = []
    current_index = 0
    current_sequence = [0]
    to_be_added = False

    for label in labels[1:]:
        current_index += 1
        if label == previous_label:
            consecutive_count += 1
            current_sequence.append(current_index)
            if consecutive_count > N:
                to_be_added = True
                sequences_list.append(current_sequence)
                languages_list.append(label)
                if label not in detected_languages:
                    detected_languages.append(label)
                consecutive_count = 0
                current_sequence = []
            else:
                if to_be_added:
                    sequences_list.append(current_sequence)
                    languages_list.append(label)
                    if label not in detected_languages:
                        detected_languages.append(label)
                    to_be_added = False
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
                    result_split = last_line.split(" : ")
                    word = result_split[0].replace(" ", "")
                    labels = result_split[1].split(") ")
                    if len(labels) > 1:
                        match = re.search(r'=> (\w+) \(([\d.]+)\)', labels[-1])
                        if match is None:
                            label = "unknown"
                        else:
                            label = match.group(1)
                            probability = float(match.group(2))
                            if probability < 0.20:
                                label = "unknown"
                    else:
                        label = result_split[1].split("(")[0].replace(" ", "")

                    word_list.append(word)
                    label_list.append(label)
                except IndexError:
                    continue

        if label_list:
            mono_or_bi, valid_groups, groups = classify_instance(label_list, 10)

            group_words_list = []
            group_language_list = []
            for group in valid_groups:
                group_words = [word_list[i] for i in group]
                group_language = label_list[group[0]]
                group_words_list.append(group_words)
                group_language_list.append(group_language)

            return {'label': mono_or_bi, 'groups': group_words_list, 'languages': group_language_list}

    except subprocess.CalledProcessError as e:
        print("Error running the script:", e)
