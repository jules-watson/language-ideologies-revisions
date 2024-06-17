"""
Some splitting algorithms to split a query response string into revision sentence and justification.

Author: Raymond Liu
Date: June 2024
"""

import re
import string

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.meteor_score import meteor_score

nltk.download('punkt')
nltk.download('wordnet')


def meteor_similarity_split(original, response, split_func_args):
    """
    Strategy: 
    1. Split the sentences of the response. Make sure colons are a split delimiter too
    2. For each sentence: calculate the similarity with the original sentence, using METEOR scores.
    3. Choose the sentence that is most similar, but not equal to the original response.
       This sentence is the revision.
    4. Anything following that is the justification.
    """
    def calculate_meteor(reference, candidate):
        """
        Calculate METEOR score similarity between reference and candidate sentences.
        """
        reference_tokens = [word_tokenize(reference)]
        candidate_tokens = word_tokenize(candidate)

        return meteor_score(reference_tokens, candidate_tokens, 
                            alpha=split_func_args["alpha"], beta=split_func_args["beta"], gamma=split_func_args["gamma"])

    sentences = sent_tokenize(response.replace(":", "."))
    strip_chars = (string.whitespace + string.punctuation).replace('.', '')

    # dict of sentence: similarity score using METEOR scores
    sentence_scores = {}
    for sentence in sentences:
        s = sentence.strip(strip_chars)
        sentence_scores[s] = calculate_meteor(original, s)

    # revision = sentence with the maximum similarity score
    revision = max((s for s in sentence_scores if s != original), key=sentence_scores.get, default=None)

    if revision:
        # Justification = substring in the response after the revision sentence
        revision_index = response.find(revision)
        justification = response[revision_index + len(revision):]

        stripped_revision = revision.strip(strip_chars)
        stripped_justification = justification.strip(strip_chars)

        return stripped_revision, stripped_justification
    else:
        return None, None



def bleu_similarity_split(original, response):
    """
    Strategy: 
    1. Split the sentences of the response. Make sure colons are a split delimiter too
    2. For each sentence: calculate the similarity with the original sentence, using BLEU scores.
    3. Choose the sentence that is most similar, but not equal to the original response.
       This sentence is the revision.
    4. Anything following that is the justification.
    """
    def calculate_bleu(reference, candidate):
        """
        Calculate BLEU score similarity between reference and candidate sentences.
        """
        reference_tokens = [word_tokenize(reference)]
        candidate_tokens = word_tokenize(candidate)

        smoothing_function = SmoothingFunction().method4

        weights = (0.25, 0.25, 0.25, 0.25) # equally weigh unigrams, bigrams, trigrams, 4-grams
        return sentence_bleu(reference_tokens, candidate_tokens, weights=weights, smoothing_function=smoothing_function)
    
    sentences = sent_tokenize(response.replace(":", "."))
    strip_chars = (string.whitespace + string.punctuation).replace('.', '')

    # dict of sentence: similarity score using BLEU scores
    sentence_scores = {}
    for sentence in sentences:
        s = sentence.strip(strip_chars)
        sentence_scores[s] = calculate_bleu(original, s)

    # revision = sentence with the maximum similarity score
    revision = max((s for s in sentence_scores if s != original), key=sentence_scores.get, default=None)

    if revision:
        # Justification = substring in the response after the revision sentence
        revision_index = response.find(revision)
        justification = response[revision_index + len(revision):]

        stripped_revision = revision.strip(strip_chars)
        stripped_justification = justification.strip(strip_chars)

        return stripped_revision, stripped_justification
    else:
        return None, None


def newline_split(original, response):
    """
    Strategy: Split on the first occurence of \n. Before that is revision, after is justification
    """
    pos = response.find('\n')

    if pos != -1:
        return response[:pos].strip(), response[pos+2:].strip()
    else:
        return None, None
    

def label_split(original, response):
    """
    Strategy: Revision is string in between the substrings 'Sentence' and 'Explanation', 
    including puncutation immediately around those substrings.
    Justification is after 'Explanation'.
    """    
    rev_match = re.search("([\*]*)Sentence([: \*]*)", response)
    just_match = re.search("([\*]*)Explanation([: \*]*)", response)
    if rev_match and just_match:
        return response[rev_match.end():just_match.start()].strip(), response[just_match.end():].strip()
    else:
        return None, None


def punc_split(original, response):
    """
    Strategy: revision sentence is in square brackets. Justification is after the ending square bracket
    """
    rev_start = response.find("[")
    rev_end = response.find("]")
    if rev_start != -1 and rev_end != -1:
        return response[rev_start+1:rev_end].strip(), response[rev_end+1:].strip()
    else:
        return None, None