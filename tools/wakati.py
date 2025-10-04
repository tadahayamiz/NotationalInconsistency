
import numpy as np

def sentence_to_wakati(sentence, vocab, translate=False):
    """
    Convert sentence that is not divided by spaces into list of each word.
    Longer word is recognized in prior.
    Words not in vocab is ignored in wakati and listed in unknown_words.

    Parameters
    ----------
    sentence: str
        Sentence that is not divided by spaces.
    vocab: array_like of str, or dict of str:any type
        List of possible words in sentence.
        If vocab is dict & translate=True, each word will be translated from key to value of vocab.
    translate: bool
        If vocab is dict & translate=True, each word will be translated from key to value of vocab.

    Returns: (wakati, unknown_words)
    -------
    wakati: list of str, or list of any type when translating
        List of words appeared in the sentence.

    unknown_words: list of str
        List of words appeared in the sentence and not in vocab.
    """

    if type(vocab) == dict and translate:
        word_to_token = vocab
    else:
        word_to_token = {w: w for w in vocab}
    max_word_len = max([len(w) for w in word_to_token.keys()])

    wakati = []
    unknown_words = []
    left_sentence = sentence

    while(len(left_sentence) > 0):
        found_word = False
        for word_len in range(max_word_len, 0, -1):
            if left_sentence[:word_len] in word_to_token:
                wakati.append(word_to_token[left_sentence[:word_len]])
                left_sentence = left_sentence[word_len:]
                found_word = True
                break
        if not found_word:
            unknown_words.append(left_sentence[:1])
            left_sentence = left_sentence[1:]

    return wakati, unknown_words
