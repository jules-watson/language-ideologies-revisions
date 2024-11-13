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

    sentence_pattern = r"(.*?(?::\n*|\n+))"  # some text followed by: a colon followed by zero or more newlines OR one or more newlines
    justification_pattern = r"(^\d+\..*$)"  # some text that starts with a number followed by a period, which generallly indicates the start of a justification
    sentence_min_length = 15  # minimum length of a sentence to be considered for revision

    # split each sentence using sentence_pattern
    split_response_sentences = []
    for sentence in response_sentences:
        split_sentence = re.findall(sentence_pattern, sentence)
        remaining_part = re.sub(sentence_pattern, "", sentence)
        if remaining_part:  # add the remaining of the sentence after the last delimiter
            split_sentence.append(remaining_part)
        split_response_sentences.extend(split_sentence)  # each split includes the delimiter at the end
    # print("Split sentences:\n", split_response_sentences)
    
    strip_chars = (string.whitespace + string.punctuation).replace('.', '').replace('!', '')  # used to strip whitespace and punctuations except periods and exclamation marks from the start and/or end of sentences

    # dict of sentence: similarity score using METEOR scores
    sentence_scores = {}
    for i in range(len(split_response_sentences)):
        s1 = split_response_sentences[i]
        # print("The current sentence is:\n", s1)
        if re.match(justification_pattern, s1):  # stop looking for revisions if it seems we have reached a justification 
            break
        s1_stripped = s1.strip(strip_chars)
        if len(s1_stripped) > sentence_min_length and '"' not in s1_stripped and s1_stripped != original: # make sure the sentence has more than 10 characters, has no double quotes, and is not the original sentence
            s1_idx = response.find(s1)
            if s1_idx == -1: # sentence not found
                print(f"Sentence '{s1}' not found.\n")
            s1_meteor_score = calculate_meteor(original, s1_stripped, split_func_args)
            sentence_scores[s1] = [s1_meteor_score, s1_idx, len(s1)]
            # print("Recorded METEOR score for:\n", s1)
            # if i < len(split_response_sentences) - 2 and re.match(r'^\d+\..*$', split_response_sentences[i + 2].strip(strip_chars)): # stop looking for revisions if a justification (which starts with a number followed by a period) starts two splits after (after the "Explanation of changes" chunk)
            #     break
            if i < len(split_response_sentences) - 1 and split_response_sentences[i+1].strip(strip_chars) != original:
                s2 = split_response_sentences[i+1]
                s2_stripped = s2.strip(strip_chars)
                if len(s2_stripped) > sentence_min_length and '"' not in s2_stripped and s2_stripped != original:
                    s2_idx = response.find(s2)
                    if s2_idx == -1: # sentence not found
                        print(f"Sentence '{s2}' not found.\n")
                    s1_s2_stripped = s1_stripped + ' ' + s2_stripped  # concatenate the current and next split
                    s1_s2_meteor_score = calculate_meteor(original, s1_s2_stripped, split_func_args)
                    sentence_scores[s1 + ' ' + s2] = [s1_s2_meteor_score, s1_idx, len(s1) + len(s2)]
                    # print("Recorded METEOR score for:\n", s2)
                if i < len(split_response_sentences) - 2 and split_response_sentences[i+2].strip(strip_chars) != original:
                    s3 = split_response_sentences[i+2]
                    s3_stripped = s3.strip(strip_chars)  # concatenate the current, next, and next-next split
                    if len(s3_stripped) > sentence_min_length and '"' not in s3 and s3_stripped != original:
                        s3_idx = response.find(s3)
                        if s3_idx == -1:
                            print(f"Sentence '{s3}' not found.\n")
                        s1_s2_s3_stripped = s1_stripped + ' ' + s2_stripped + ' ' + s3_stripped
                        s1_s2_s3_meteor_score = calculate_meteor(original, s1_s2_s3_stripped, split_func_args)
                        sentence_scores[s1 + ' ' + s2 + ' ' + s3] = [s1_s2_s3_meteor_score, s1_idx, len(s1) + len(s2) + len(s3)]
                        # print("Recorded METEOR score for:\n", s3)

    # revision = sentence with the maximum similarity score
    revision = max(sentence_scores, key=lambda s: sentence_scores[s][0], default=None)

    if revision:
        # Justification = substring in the response after the revision sentence
        # revision_index = response.find(revision)
        # if revision_index == -1: # revision not found
        #     print(f"Revision '{revision}' not found. Make sure to check the corresponding justification column!\n")
        revision_idx = sentence_scores[revision][1]
        revision_len = sentence_scores[revision][2]
        justification = response[revision_idx + revision_len:]

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