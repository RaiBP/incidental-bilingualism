from datasets import load_dataset
import json
from tqdm import tqdm


#Specify the dataset name
dataset_name = "RaiBP/openwebtext2-first-30-chunks-lang-detect-raw-output"

# Load the dataset
bilingual_dataset = load_dataset(dataset_name, data_dir='bilingual')

dataset = bilingual_dataset["train"]
n_examples = len(dataset)
keys_dict = {}
for document in tqdm(dataset, total=n_examples):

    instance_labels = document["instance_labels"]
    instance_languages = document["instance_languages"]

    for languages in instance_languages:
        unique_languages = list(set(languages))
        lang_key = "-".join(sorted(unique_languages))
        if lang_key not in keys_dict.keys():
            keys_dict[lang_key] = 1
        else:
            keys_dict[lang_key] += 1

english_keys_list = []  # keys where "en" is present
non_english_keys_list = []  # keys where "en" is not present
for key in keys_dict.keys():
    key_list = key.split('-')
    if "en" in key_list:
        english_keys_list.append(key_list)
    else:
        non_english_keys_list.append(key_list)

# more than two languages, none of them English
nen_multi_count = 0
# one language, one of the following: de, fr, es, pt, it, nl
lang_mono_count = {'de': 0, 'fr': 0, 'es': 0, 'pt': 0, 'it': 0, 'nl': 0}
# one language, not one of the following: de, fr, es, pt, it, nl
other_mono_count = 0
# two languages, none of them English
nen_bi_count = 0
for key in non_english_keys_list:
    if len(key) > 2:
        nen_multi_count += keys_dict['-'.join(key)]
    elif len(key) == 2:
        nen_bi_count += keys_dict['-'.join(key)]
    elif len(key) == 1:
        nen_lang = key[0]
        if nen_lang in lang_mono_count.keys():
            lang_mono_count[nen_lang] += keys_dict[nen_lang]
        else:
            other_mono_count += keys_dict[nen_lang]

# more than two languages, at least one of them English
english_multi_count = 0
# one language, English
english_mono_count = 0
for key in english_keys_list:
    if len(key) == 1 and key[0] == 'en':
        english_mono_count += keys_dict[key[0]]
    if len(key) > 2:
        english_multi_count += keys_dict['-'.join(key)]

# two languages, one of them English, the other one not one of the following: de, fr, es, pt, it, nl
other_bi_count = 0
# two languages, one of them English, the other one of the following: de, fr, es, pt, it, nl
lang_bi_count = {'de': 0, 'fr': 0, 'es': 0, 'pt': 0, 'it': 0, 'nl': 0}
for key in english_keys_list:
    if len(key) == 2:
        nen_lang = key[0] if key[1] == 'en' else key[1]
        if nen_lang in lang_bi_count.keys():
            lang_bi_count[nen_lang] += keys_dict['-'.join(key)]
        else:
            other_bi_count += keys_dict['-'.join(key)]

# Save the counts for monolingual
counts_dict_monolingual = {"en": english_mono_count}
for lang in lang_mono_count.keys():
    counts_dict_monolingual[lang] = lang_mono_count[lang]
counts_dict_monolingual["other"] = other_mono_count

with open('monolingual_counts.json', 'w') as json_file:
    json.dump(counts_dict_monolingual, json_file)

# Save the counts for bilingual
counts_dict_bilingual = {}
for lang in lang_bi_count.keys():
    counts_dict_bilingual[lang] = lang_bi_count[lang]
counts_dict_bilingual["other"] = other_bi_count + nen_bi_count + english_multi_count + nen_multi_count

with open('bilingual_counts.json', 'w') as json_file:
    json.dump(counts_dict_bilingual, json_file)
