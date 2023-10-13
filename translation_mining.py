import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
import nltk
from collections import Counter
import Levenshtein
from ftlangdetect import detect

# Check if 'punkt' is already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    # 'punkt' data is not downloaded, so download it
    nltk.download('punkt')



# Load the mBERT tokenizer
multi_tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")


def get_similarity_score(embeddings1, embeddings2):
    # Calculate cosine similarity
    similarity_matrix = torch.matmul(
        F.normalize(embeddings1, p=2),
        F.normalize(embeddings2, p=2).transpose(0, 1)
    )
    return similarity_matrix


def get_embeddings(sentences):
    tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
    model = BertModel.from_pretrained("setu4993/LaBSE")

    model = model.eval()

    inputs = tokenizer(sentences, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.pooler_output


def detect_translations(sentences_embedded, sentences_primary, threshold=0.6):
    embeddings_embedded = get_embeddings(sentences_embedded)
    embeddings_primary = get_embeddings(sentences_primary)

    similarity_matrix = get_similarity_score(embeddings_embedded, embeddings_primary)

    translation_pairs = []
    for i in range(len(sentences_embedded)):
        for j in range(len(sentences_primary)):
            if similarity_matrix[i][j] > threshold:
                translation_pairs.append((sentences_embedded[i], sentences_primary[j]))

    return translation_pairs


def sentence_breaker(words, labels):
    text = " ".join(words)

    sentences = sent_tokenize(text)

    sentence_list = []
    sentence_label_list = []

    for sentence in sentences:
        # Tokenize the sentence into words
        sentence_words = sentence.split()

        # Get the labels for words in the sentence
        sentence_labels = [labels[words.index(word)] for word in sentence_words if word in words]

        # Count the occurrences of each label in the sentence
        label_counts = Counter(sentence_labels)

        # Find the label with the highest count (majority voting)
        majority_label = label_counts.most_common(1)[0][0]

        sentence_list.append(sentence)
        sentence_label_list.append(majority_label)

    return sentence_list, sentence_label_list


def extract_embedded_and_primary_sentences(sentences, sentence_labels):
    # Count the occurrences of each label in the entire text
    label_counts = Counter(sentence_labels)

    # Find the label with the most sentences
    embedded_label = label_counts.most_common(1)[0][0]

    # Find the label with the second most sentences
    primary_label = label_counts.most_common(2)[1][0]

    # Initialize variables to store primary and secondary sentences
    embedded_sentences = []
    primary_sentences = []

    # Iterate through the sentences and categorize them
    for sentence, label in zip(sentences, sentence_labels):
        if label == embedded_label:
            embedded_sentences.append(sentence)
        elif label == primary_label:
            primary_sentences.append(sentence)

    return embedded_sentences, primary_sentences, embedded_label, primary_label


def get_min_edit_distance_filter(sentence_1, sentence_2, min_edit_distance, min_edit_distance_ratio):
    # Calculate the edit distance between sentences
    edit_distance = Levenshtein.distance(sentence_1, sentence_2)

    # Calculate the edit distance ratio
    max_length = max(len(sentence_1), len(sentence_2))
    edit_distance_ratio = edit_distance / max_length

    # Check edit distance conditions
    return edit_distance >= min_edit_distance and edit_distance_ratio >= min_edit_distance_ratio


def get_language_detection_filter(sentence_1, sentence_2):
    lang1 = detect(text=sentence_1, low_memory=False)["lang"]
    lang2 = detect(text=sentence_2, low_memory=False)["lang"]
    return lang1 != lang2


def apply_filters(sentence_1, sentence_2, min_length=3, max_length=200, max_ratio=2.0, min_edit_distance=2,
                  min_edit_distance_ratio=0.1):
    global multi_tokenizer
    # Tokenize sentences using the mBERT tokenizer
    tokens_1 = multi_tokenizer.tokenize(sentence_1)
    tokens_2 = multi_tokenizer.tokenize(sentence_2)

    len_tokens_1 = len(tokens_1)
    len_tokens_2 = len(tokens_2)

    token_length_filter = min_length <= len_tokens_1 <= max_length and min_length <= len_tokens_2 <= max_length
    max_length_ratio_filter = max(len_tokens_1 / len_tokens_2, len_tokens_2 / len_tokens_1) <= max_ratio
    min_edit_distance_filter = get_min_edit_distance_filter(sentence_1, sentence_2, min_edit_distance,
                                                            min_edit_distance_ratio)
    language_detection_filter = get_language_detection_filter(sentence_1, sentence_2)
    return token_length_filter and max_length_ratio_filter and min_edit_distance_filter and language_detection_filter
