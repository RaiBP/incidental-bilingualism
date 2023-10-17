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


def merge_ambiguous_groups(ambiguous_groups):
    """
    We merge ambiguous groups when their indices are consecutive. It is more likely that they are part of the same
    language than not.
    :param ambiguous_groups: A list of ambiguous groups.
    :return:
    """
    output = []
    current_group = []

    for group in ambiguous_groups:
        if not current_group:
            current_group = group
        elif current_group[-1] + 1 == group[0]:
            current_group += group
        else:
            output.append(current_group)
            current_group = group

    if current_group:
        output.append(current_group)

    return output


def unambiguate_groups(ambiguous_groups, words, labels):
    ambiguous_groups_labels = []
    ambiguous_groups = merge_ambiguous_groups(ambiguous_groups)

    for ambiguous_group in ambiguous_groups:
        ambiguous_text = " ".join([words[i] for i in ambiguous_group])
        result = detect(text=ambiguous_text, low_memory=False)
        ambiguous_groups_labels.append(result["lang"])
    corrected_labels = replace_ambiguous_labels(labels, ambiguous_groups, ambiguous_groups_labels)
    return obtain_groups_from_labels(corrected_labels), corrected_labels


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


def classify_instance(labels, words, probs, consecutive_threshold=10, probability_threshold=0.6):
    """
    Classify an instance as monolingual or bilingual. If there are two or more consecutive sequences (groups) of more
    than N words in different languages, the instance is bilingual, if not it is monolingual.

    :param labels: A list of language labels returned by CoSwID. One for each word in the instance.
    :param words: List of words in the instance.
    :param probs: List of the probability of each word being in the language specified by the label.
    :param consecutive_threshold: The minimum number of consecutive words for it to be considered a valid sequence.
        Default is 10.
    :param probability_threshold: The minimum average probability of the words in a group for it to not be considered an
        ambiguous group.

    :return: A tuple containing the classification ("bi" or "mono") and the valid groups. Note that groups are lists of
    indices of words in the instance.
    """
    unique_labels = list(set(labels))
    if len(unique_labels) == 1 and unique_labels[0] != "unknown":
        return "mono", [list(range(len(labels)))]
    groups = obtain_groups_from_labels(labels)
    average_prob = [sum([probs[i] for i in group]) / len(group) for group in groups]
    ambiguous_groups = [group for index, group in enumerate(groups) if average_prob[index] < probability_threshold]
    if len(ambiguous_groups) > 0:
        groups, labels = unambiguate_groups(ambiguous_groups, words, labels)

    valid_groups_indices = [index for index, group in enumerate(groups) if len(group) > consecutive_threshold]

    valid_groups = [groups[i] for i in valid_groups_indices]
    valid_groups_tags = [labels[group[0]] for group in valid_groups]

    label = "bi" if len(set(valid_groups_tags)) > 1 else "mono"
    return label, valid_groups


def format_label(label):
    if len(label) == 3:
        try:
            language = languages.get(part3=label)
            return language.part1 if language else "unknown"
        except AttributeError:
            return "unknown"
    else:
        return label


def detect_code_switching(instance, coswid_path, coswid_model, consecutive_threshold):
    try:
        coswid_output = code_switching_language_detector(coswid_model, coswid_path, instance)

        label_list, prob_list, word_list = extract_coswid_results(coswid_output)

        if label_list:
            mono_or_bi, groups = classify_instance(label_list, word_list, prob_list, consecutive_threshold)

            group_words_list = []
            group_language_list = []
            for group in groups:
                group_words = [word_list[i] for i in group]
                group_language = label_list[group[0]]
                group_words_list.append(group_words)
                group_language_list.append(group_language)

            return mono_or_bi, word_list, label_list, group_words_list, group_language_list

    except subprocess.CalledProcessError as e:
        print("Error running the script:", e)


def extract_coswid_results(coswid_output):
    # Define the regular expression pattern
    pattern = r'Thresholding :'
    pattern2 = r'\)\s*\n\[\d+\] : '
    # Split the string based on the pattern
    first_split = re.split(pattern, coswid_output)
    # we don't need the first part
    first_split = first_split[1:]
    parts = []
    for part in first_split:
        parts.append(re.split(pattern2, part)[0])

    word_list = []
    label_list = []
    prob_list = []
    # Loop through the parts and print them
    for index, part in enumerate(parts):
        # Split the part into lines and select the last line
        lines = part.strip().splitlines()
        if lines:
            last_line = lines[-1]
            try:
                pattern = r'(.+?)\s*:\s*(\w+)\s*\((\d+\.\d+)'
                if ' => ' in last_line:
                    # CoSwID outputs multiple labels
                    last_line_single_label = last_line.split(" : ")[0] + " : " + last_line.split(' => ')[1]
                else:
                    # CoSwID outputs a single label
                    last_line_single_label = last_line
                match = re.search(pattern, last_line_single_label)

                try:
                    word = match.group(1)
                    label = match.group(2)
                    prob = float(match.group(3))
                except AttributeError:
                    continue

                word_list.append(word)
                label_list.append(format_label(label))
                prob_list.append(prob)
            except IndexError:
                continue
    return label_list, prob_list, word_list


def code_switching_language_detector(coswid_model, coswid_path, instance):
    coswid_arguments = ["-m", coswid_model, "-t", instance, "-c", "2", "-f", "0", "-g", "0.1", "-v", "dico"]
    process = subprocess.Popen(["python3", coswid_path] + coswid_arguments, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return stdout
