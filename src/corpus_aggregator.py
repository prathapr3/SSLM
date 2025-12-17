import sanskrit_tokenizer

def aggregate_corpus(corpus_files):
    """
    Aggregates multiple Sanskrit text corpora into a single corpus.

    Args:
        corpus_files (list of str): List of file paths to the Sanskrit text corpora.

    Returns:
        str: The aggregated corpus as a single string.
    """
    aggregated_corpus = ""

    for file_path in corpus_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            aggregated_corpus += text + "\n"

    return aggregated_corpus.strip()

def tokenize_corpus(corpus, vocab):
    """
    Tokenizes the given Sanskrit corpus using the provided vocabulary.

    Args:
        corpus (str): The Sanskrit text corpus to tokenize.
        vocab (dict): A dictionary mapping strings to integer IDs.

    Returns:
        list of int: The tokenized corpus as a list of integer IDs.
    """
    tokenizer = sanskrit_tokenizer.SimpleSanskritTokenizer(vocab)
    tokenized_ids = tokenizer.encode(corpus)
    return tokenized_ids

def detokenize_corpus(token_ids, vocab):
    """
    Detokenizes the given list of token IDs back into Sanskrit text.

    Args:
        token_ids (list of int): The list of token IDs to detokenize.
        vocab (dict): A dictionary mapping strings to integer IDs.

    Returns:
        str: The detokenized Sanskrit text.
    """
    tokenizer = sanskrit_tokenizer.SimpleSanskritTokenizer(vocab)
    detokenized_text = tokenizer.decode(token_ids)
    return detokenized_text

def save_tokenized_corpus(token_ids, output_file):
    """
    Saves the tokenized corpus to a file.

    Args:
        token_ids (list of int): The list of token IDs to save.
        output_file (str): The file path to save the tokenized corpus.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(" ".join(map(str, token_ids)))

def load_tokenized_corpus(input_file):
    """
    Loads a tokenized corpus from a file.

    Args:
        input_file (str): The file path to load the tokenized corpus from.

    Returns:
        list of int: The loaded list of token IDs.
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
        token_ids = list(map(int, content.split()))
    return token_ids
