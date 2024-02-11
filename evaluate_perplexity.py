import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import argparse

# Set the seed for reproducibility
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
langs = ["en", "de", "es", "fr", "it", "pt", "nl"]


def sample_random_examples(num_examples, dataset):
    random_numbers = list(np.random.randint(0, len(dataset), num_examples))
    examples = []
    for i in tqdm(random_numbers, desc="Sampling examples"):
        examples.append(dataset[int(i)]["text"])
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        "-m",
        type=int,
        help="Model to evaluate Possible values: 0, 1, 2, 3, 4 corresponding to bilingual, translation, non-english, full and gpt2",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=2000,
        help="Number of examples to evaluate at each run",
    )
    parser.add_argument(
        "--number_of_runs",
        "-n",
        type=int,
        default=10,
        help="Number of runs to average the perplexity",
    )
    parser.add_argument("--language", "-l", type=str, help="Language to evaluate")
    # Load the model
    models = [
        "RaiBP/gpt2-openwebtext2-first-30-chunks-ablation-bilingual",
        "RaiBP/gpt2-openwebtext2-first-30-chunks-ablation-translation",
        "RaiBP/gpt2-openwebtext2-first-30-chunks-ablation-non-english",
        "RaiBP/gpt2-openwebtext2-first-30-chunks-ablation-full",
        "gpt2",
    ]
    args = parser.parse_args()

    model_name = models[args.model_id]
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lang = args.language
    ppl_list = []
    dataset = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train")
    for try_idx in tqdm(range(args.number_of_runs), desc="Processing batches"):
        examples = sample_random_examples(args.batch_size, dataset)
        encodings = tokenizer("\n\n\n".join(examples), return_tensors="pt")
        max_length = model.config.n_positions
        stride = 1024
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride), "Processing sequences"):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = (
                end_loc - prev_end_loc
            )  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        ppl_item = ppl.item()
        ppl_list.append(ppl_item)
        print(
            f"Model {model_name}: Perplexity for {lang} at run {try_idx}: ",
            ppl_item,
        )
    print("Perplexity list: ", ppl_list)
    print(
        f"Model {model_name}: Perplexity for {lang}: {np.mean(ppl_list)} +/- {np.std(ppl_list)}"
    )
