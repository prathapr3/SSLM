import corpus_aggregator

if __name__ == "__main__":
    # Example usage of the corpus aggregator and tokenizer

    # Step 1: Aggregate corpus from multiple files
    corpus_files = ['data/text1.txt', 'data/text2.txt', 'data/text3.txt']
    aggregated_corpus = corpus_aggregator.aggregate_corpus(corpus_files)

    # Step 2: Define a simple vocabulary for tokenization
    vocab = {
        'अ': 1,
        'आ': 2,
        'इ': 3,
        'ई': 4,
        'उ': 5,
        'ऊ': 6,
        'क': 7,
        'ख': 8,
        'ग': 9,
        'घ': 10,
        # Add more vocabulary as needed
    }

    # Step 3: Tokenize the aggregated corpus
    tokenized_ids = corpus_aggregator.tokenize_corpus(aggregated_corpus, vocab)

    # Step 4: Save the tokenized corpus to a file
    output_file = 'data/tokenized_corpus.txt'
    corpus_aggregator.save_tokenized_corpus(tokenized_ids, output_file)

    # Step 5: Load the tokenized corpus from the file
    loaded_token_ids = corpus_aggregator.load_tokenized_corpus(output_file)

    # Step 6: Detokenize the loaded token IDs back into text
    detokenized_text = corpus_aggregator.detokenize_corpus(loaded_token_ids, vocab)

    print("Detokenized Text:")
    print(detokenized_text)

