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


def calculate_meteor(reference, candidate, split_func_args):
    """
    Calculate METEOR score similarity between reference and candidate sentences.
    """
    reference_tokens = [word_tokenize(reference)]
    candidate_tokens = word_tokenize(candidate)

    return meteor_score(reference_tokens, candidate_tokens, 
                        alpha=split_func_args["alpha"], beta=split_func_args["beta"], gamma=split_func_args["gamma"])


def meteor_heuristic_split(original, response, split_func_args):
    """
    Strategy: 
    1. Tokenize the response into sentences without using any split delimiters. 
    2. Split each tokenized sentence using newlines and colons as delimiters.
    3. For each split: calculate the similarity with the original sentence, using METEOR scores.
    4. For each group of 2 or 3 consecutive splits: concatenate them and calculate the METEOR similarity with the original sentence.
    5. Choose the individual or concatenated split that is the most similar, but not equal to the original response.
       This sentence is the revision.
    6. Anything following that is the justification.
    """
    response_sentences = sent_tokenize(response)
    # print("Tokenized sentences:\n", response_sentences)   
    split_response_sentences = []
    pattern = r"(.*?(?::\n*|\n+))"  # some text followed by: a colon followed by zero or more newlines OR one or more newlines
    for sentence in response_sentences:
        split_sentence = re.findall(pattern, sentence)
        remaining_part = re.sub(pattern, "", sentence)
        if remaining_part:  # add the remaining of the sentence after the last delimiter
            split_sentence.append(remaining_part)
        split_response_sentences.extend(split_sentence)  # each split includes the delimiter at the end
    # print("Split sentences:\n", split_response_sentences)
    
    strip_chars = (string.whitespace + string.punctuation).replace('.', '').replace('!', '')  # used to strip whitespace and punctuations except periods and exclamation marks from the start and/or end of sentences

    # dict of sentence: similarity score using METEOR scores
    sentence_scores = {}
    for i in range(len(split_response_sentences)):
        s1 = split_response_sentences[i].strip(strip_chars)
        # print("The current sentence is:\n", s1)
        if len(s1) > 10 and '"' not in s1 and s1 != original: # make sure the sentence has more than 10 characters, has no double quotes, and is not the original sentence!
            sentence_scores[s1] = calculate_meteor(original, s1, split_func_args)
            # print("Recorded METEOR score for:\n", s1)
            if i < len(split_response_sentences) - 2 and re.match(r'^\d+\..*$', split_response_sentences[i + 2].strip(strip_chars)): # stop looking for revisions if a justification (which starts with a number followed by a period) starts two splits after (after the "Explanation of changes" chunk)
                break
            if i < len(split_response_sentences) - 1 and split_response_sentences[i+1].strip(strip_chars) != original:
                s2 = split_response_sentences[i].lstrip(strip_chars) + ' ' + split_response_sentences[i+1].rstrip(strip_chars)  # concatenate the current and next split
                if len(s2) > 10 and '"' not in s2:
                    sentence_scores[s2] = calculate_meteor(original, s2, split_func_args)
                    # print("Recorded METEOR score for:\n", s2)
                if i < len(split_response_sentences) - 2 and split_response_sentences[i+2].strip(strip_chars) != original:
                    s3 = split_response_sentences[i].lstrip(strip_chars) + ' ' + split_response_sentences[i+1] + ' ' + split_response_sentences[i+2].rstrip(strip_chars)  # concatenate the current, next, and next-next split
                    if len(s3) > 10 and '"' not in s3:
                        sentence_scores[s3] = calculate_meteor(original, s3, split_func_args)
                        # print("Recorded METEOR score for:\n", s3)

    # revision = sentence with the maximum similarity score
    revision = max((s for s in sentence_scores if s != original), key=sentence_scores.get, default=None)

    if revision:
        # Justification = substring in the response after the revision sentence
        revision_index = response.find(revision)
        print("Make sure to check the corresponding justification column for the following revisions, if applicable!")
        if revision_index == -1: # revision not found
            print(f"Revision '{revision}'")
        justification = response[revision_index + len(revision):]

        stripped_revision = revision.strip(strip_chars)
        stripped_justification = justification.strip(strip_chars)

        return stripped_revision, stripped_justification
    else:
        return None, None
    
def meteor_similarity_split(original, response, split_func_args):
    """
    Strategy: 
    1. Split the sentences of the response. Make sure colons are a split delimiter too
    2. For each sentence: calculate the similarity with the original sentence, using METEOR scores.
    3. Choose the sentence that is most similar, but not equal to the original response.
       This sentence is the revision.
    4. Anything following that is the justification.
    """
    sentences = sent_tokenize(response.replace(":", "."))
    strip_chars = (string.whitespace + string.punctuation).replace('.', '')

    # dict of sentence: similarity score using METEOR scores
    sentence_scores = {}
    for sentence in sentences:
        s = sentence.strip(strip_chars)
        if len(s) > 10: # make sure the sentence has more than 10 characters!
            sentence_scores[s] = calculate_meteor(original, s, split_func_args)

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