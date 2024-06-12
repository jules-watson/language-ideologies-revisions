"""
Some splitting algorithms to split a query response string into revision sentence and justification.
"""

import re
import string

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.meteor_score import meteor_score

nltk.download('punkt')
nltk.download('wordnet')

def calculate_meteor(reference, candidate):
    """
    Calculate METEOR score similarity between reference and candidate sentences.
    """
    reference_tokens = [word_tokenize(reference)]
    candidate_tokens = word_tokenize(candidate)

    return meteor_score(reference_tokens, candidate_tokens, alpha=0.5, beta=2.0, gamma=0.3)



def calculate_bleu(reference, candidate):
    """
    Calculate BLEU score similarity between reference and candidate sentences.
    """
    reference_tokens = [word_tokenize(reference)]
    candidate_tokens = word_tokenize(candidate)

    smoothing_function = SmoothingFunction().method4

    weights = (0.25, 0.25, 0.25, 0.25) # equally weigh unigrams, bigrams, trigrams, 4-grams
    return sentence_bleu(reference_tokens, candidate_tokens, weights=weights, smoothing_function=smoothing_function)



def split_using_similarity(original, response, similarity_algo=calculate_meteor):
    """
    Strategy: 
    1. Split the sentences of the response. Make sure colons are a split delimiter too
    2. For each sentence: calculate the similarity with the original sentence, using similarity_algo.
    3. Choose the sentence that is most similar, but not equal to the original response.
       This sentence is the revision.
    4. Anything following that is the justification.
    """
    sentences = sent_tokenize(response.replace(":", "."))
    strip_chars = (string.whitespace + string.punctuation).replace('.', '')

    # dict of sentence: similarity score using similarity_algo
    sentence_scores = {}
    for sentence in sentences:
        s = sentence.strip(strip_chars)
        sentence_scores[s] = similarity_algo(original, s)

    # revision = sentence with the maximum similarity score
    revision = max((s for s in sentence_scores if s != original), key=sentence_scores.get, default=None)

    # Step 4: Anything following the revision is the justification.
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



"""
Unit tests
"""

import unittest

class SplitUsingSimilarity(unittest.TestCase):

    def test(self):
        original = "Alex has passed many bills while working as a congressperson."
        response = """Revised sentence: During his tenure as a congressperson, Alex successfully passed numerous bills.

Explanation of changes:
1. **Replacing ""has passed"" with ""successfully passed""**: The addition of ""successfully"" emphasizes the achievement, implying effectiveness and positive outcomes, rather than merely the action of passing bills.

2. **Using ""numerous"" instead of ""many""**: ""Numerous"" often conveys a slightly more formal tone than ""many"" and can imply a greater quantity or abundance, enhancing the impression of Alex's productivity and accomplishments.

3. **Introducing ""During his tenure""**: This phrase specifies the period during which Alex passed the bills and adds a professional nuance, suggesting a sustained performance over time rather than scattered individual achievements. 

These modifications collectively enhance the clarity, tone, and impact of the original sentence, providing a more precise and polished description of Alex's accomplishments."""

        revision, justification = split_using_similarity(original, response)
        correct_revision = "During his tenure as a congressperson, Alex successfully passed numerous bills."
        correct_justification = """Explanation of changes:
1. **Replacing ""has passed"" with ""successfully passed""**: The addition of ""successfully"" emphasizes the achievement, implying effectiveness and positive outcomes, rather than merely the action of passing bills.

2. **Using ""numerous"" instead of ""many""**: ""Numerous"" often conveys a slightly more formal tone than ""many"" and can imply a greater quantity or abundance, enhancing the impression of Alex's productivity and accomplishments.

3. **Introducing ""During his tenure""**: This phrase specifies the period during which Alex passed the bills and adds a professional nuance, suggesting a sustained performance over time rather than scattered individual achievements. 

These modifications collectively enhance the clarity, tone, and impact of the original sentence, providing a more precise and polished description of Alex's accomplishments."""

        self.assertEqual(revision, correct_revision)
        self.assertEqual(justification, correct_justification)


if __name__=="__main__":
    unittest.main()