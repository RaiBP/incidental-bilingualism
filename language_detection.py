import subprocess
import re
from ftlangdetect import detect
from iso639 import languages


def replace_ambiguous_labels(labels, ambiguous_groups, ambiguous_groups_corrected_labels):
    corrected_labels = list(labels)  # Create a copy of the original labels

    # Iterate through the ambiguous groups and update the labels
    for ambiguous_group, corrected_label in zip(ambiguous_groups, ambiguous_groups_corrected_labels):
        for index in ambiguous_group:
            corrected_labels[index] = corrected_label

    return corrected_labels


def unambiguate_groups(ambiguous_groups, words, labels):
    ambiguous_groups_labels = []
    for ambiguous_group in ambiguous_groups:
        ambiguous_text = " ".join([words[i] for i in ambiguous_group])
        result = detect(text=ambiguous_text, low_memory=False)
        ambiguous_groups_labels.append(result["lang"])
    corrected_labels = replace_ambiguous_labels(labels, ambiguous_groups, ambiguous_groups_labels)
    return obtain_groups_from_labels(corrected_labels)


def obtain_groups_from_labels(labels):
    if len(set(labels)) == 1:
        return [list(range(len(labels)))]

    groups = []
    current_group = []

    for i, label in enumerate(labels):
        if i == 0 or label == labels[i - 1]:
            # we form our group with consecutive labels
            current_group.append(i)
        else:
            # if the current label is different from the previous one, we start a new group
            groups.append(current_group)
            current_group = [i]

    # we add the last group
    groups.append(current_group)
    return groups


def obtain_groups_and_ambiguous_groups_from_labels(labels, ambiguous_indices, ambiguous_threshold):
    groups = []
    ambiguous_groups = []
    current_group = []
    ambiguous_count = 0

    for i, label in enumerate(labels):
        if i == 0 or label == labels[i - 1]:
            # we form our group with consecutive labels
            current_group.append(i)
            if i in ambiguous_indices:
                # if our current word is ambiguous, we increment the ambiguous count
                ambiguous_count += 1
        else:
            # if the current label is different from the previous one, we start a new group
            groups.append(current_group)
            if ambiguous_count > ambiguous_threshold:
                # if the ambiguous count is greater than the threshold, we add the group to the ambiguous groups
                ambiguous_groups.append(current_group)
            # we reset the ambiguous count and start a new group
            ambiguous_count = 0
            current_group = [i]

    # we add the last group
    groups.append(current_group)
    if ambiguous_count > ambiguous_threshold:
        ambiguous_groups.append(current_group)
    return groups, ambiguous_groups


def classify_instance(labels, words, ambiguous_indices, consecutive_threshold=10, ambiguous_threshold=5):
    """
    Classify an instance as monolingual or bilingual. If there are two or more consecutive sequences (groups) of more
    than N words in different languages, the instance is bilingual, if not it is monolingual.
    :param labels: A list of language labels returned by CoSwID. One for each word in the instance.
    :param words: List of words in the instance.
    :param ambiguous_indices: The indices of the ambiguous words in the instance.
    :param consecutive_threshold: The minimum number of consecutive words for it to be considered a valid sequence.
    Default is 10.
    :param ambiguous_threshold: The minimum number of ambiguous words for a sequence to be considered ambiguous.
    :return: A tuple containing the classification ("bi" or "mono"), the valid groups, and all the groups. Note that
    groups are lists of indices of words in the instance.
    """
    if len(set(labels)) == 1:
        return "mono", [list(range(len(labels)))], [list(range(len(labels)))]
    groups, ambiguous_groups = obtain_groups_and_ambiguous_groups_from_labels(labels, ambiguous_indices,
                                                                              ambiguous_threshold)

    if len(ambiguous_groups) > 0:
        groups = unambiguate_groups(ambiguous_groups, words, labels)

    valid_groups_indices = [index for index, group in enumerate(groups) if len(group) > consecutive_threshold]

    valid_groups = [groups[i] for i in valid_groups_indices]
    valid_groups_labels = [labels[group[0]] for group in valid_groups]
    return "bi" if len(set(valid_groups_labels)) > 1 else "mono", valid_groups, groups


def format_label(label):
    if len(label) == 3:
        try:
            language = languages.get(part3=label)
            return language.part1 if language else "unknown"
        except AttributeError:
            return "unknown"
    else:
        return label


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
        ambiguous_indices = []

        # Loop through the parts and print them
        for index, part in enumerate(parts):
            # Split the part into lines and select the last line
            lines = part.strip().splitlines()
            if lines:
                last_line = lines[-1]
                try:
                    result_split = last_line.split(" : ")
                    word = result_split[0].replace(" ", "")
                    labels = result_split[1].split(") ")
                    if len(labels) > 1:
                        # if more than one label is returned by CoSwID, it means that the word is ambiguous
                        ambiguous_indices.append(index)
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
                    label_list.append(format_label(label))
                except IndexError:
                    continue

        if label_list:
            mono_or_bi, valid_groups, groups = classify_instance(label_list, word_list, ambiguous_indices, 10, 2)

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
