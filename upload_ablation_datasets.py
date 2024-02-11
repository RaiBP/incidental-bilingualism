from datasets import load_dataset, concatenate_datasets
import random


random.seed(42)

translation_dataset = load_dataset('RaiBP/openwebtext2-first-30-chunks-translation-examples')
bilingual_dataset = load_dataset('RaiBP/openwebtext2-first-30-chunks-bilingual-examples')
non_english_dataset = load_dataset('RaiBP/openwebtext2-first-30-chunks-nonenglish-examples')
english_only_dataset = load_dataset('RaiBP/openwebtext2-first-30-chunks-english-only-examples')


def select_random_examples(dataset, n_examples):
    total_examples = len(dataset)

    # Ensure that n_examples is not greater than the total number of examples
    if n_examples > total_examples:
        raise ValueError("n_examples should not be greater than the total number of examples in the dataset")

    # Use random.sample to select n_examples at random
    random_indices = random.sample(range(total_examples), n_examples)

    # Extract the randomly selected examples
    selected_examples = dataset.select(random_indices)

    return selected_examples


english_only_examples = english_only_dataset.num_rows['data']
non_english_examples = non_english_dataset.num_rows['train']
bilingual_examples = bilingual_dataset.num_rows['train']
translation_examples = translation_dataset.num_rows['train']

english_only_ablation_examples = english_only_examples - non_english_examples
non_english_ablation_examples = non_english_examples - bilingual_examples
bilingual_ablation_examples = bilingual_examples - translation_examples

english_only_ablation = select_random_examples(english_only_dataset['data'], english_only_ablation_examples)
non_english_ablation = select_random_examples(non_english_dataset['train'], non_english_ablation_examples)
bilingual_ablation = select_random_examples(bilingual_dataset['train'], bilingual_ablation_examples)

bilingual_ablated_dataset = concatenate_datasets([english_only_ablation, non_english_dataset['train']])
translation_ablated_dataset = concatenate_datasets(
    [english_only_ablation, non_english_ablation, bilingual_dataset['train']])
full_dataset = concatenate_datasets(
    [english_only_ablation, non_english_ablation, bilingual_ablation, translation_dataset['train']])

full_dataset.push_to_hub("RaiBP/openwebtext2-first-30-chunks-ablation-full")
translation_ablated_dataset.push_to_hub("RaiBP/openwebtext2-first-30-chunks-ablation-translation")
bilingual_ablated_dataset.push_to_hub("RaiBP/openwebtext2-first-30-chunks-ablation-bilingual")
english_only_dataset['data'].push_to_hub("RaiBP/openwebtext2-first-30-chunks-ablation-non-english")

