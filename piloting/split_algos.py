"""
Some splitting algorithms to split a query response string into revision sentence and justification.
"""

import re
import string

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

def calculate_bleu(reference, candidate):
    """
    Calculate BLEU score between reference and candidate sentences
    """
    reference_tokens = [word_tokenize(reference)]
    candidate_tokens = word_tokenize(candidate)

    smoothing_function = SmoothingFunction().method1

    weights = (0.25, 0.25, 0.25, 0.25) # equally weigh unigrams, bigrams, trigrams, 4-grams
    return sentence_bleu(reference_tokens, candidate_tokens, weights=weights, smoothing_function=smoothing_function)


def bleu_split(original, response):
    """
    Strategy: 
    1. Split the sentences of the response. Make sure colons are a split delimiter too
    2. For each sentence: calculate the similarity of the BLEU score with the original sentence.
    3. Choose the sentence that is most similar, but not equal to the original response.
       This sentence is the revision.
    4. Anything following that is the justification.
    """
    sentences = sent_tokenize(response.replace(":", "."))

    # dict of sentence: BLEU score
    sentence_scores = {sentence: calculate_bleu(original, sentence) for sentence in sentences}

    # revision = sentence with the maximum bleu score
    revision = max((s for s in sentence_scores if s != original), key=sentence_scores.get, default=None)

    # Step 4: Anything following the revision is the justification.
    if revision:
        # Join the later sentences after the revision
        revision_index = sentences.index(revision)
        justification = " ".join(sentences[revision_index + 1:])

        # alternatively: find the substring in the response after the revision sentence
        # revision_index = response.find(revision)
        # justification = response[revision_index + len(revision):]
        punctuation_without_period = ''.join([char for char in string.punctuation if char != '.'])
        stripped_revision = revision.strip(punctuation_without_period)

        if stripped_revision == "This alteration provides a slightly more formal and polished tone, augmenting the impact of Alex's handsomeness in the photo.":
            print(sentences)
            print(f"Revised sentence is: {revision}")
            print(f"Reference sentence is: {original}")
            print(f"BLEU score for incorrect: {sentence_scores[revision]}")
            correct = "Alex appeared strikingly handsome in the photograph."
            print(f"BLEU score for correct: {sentence_scores[correct]}")
        return stripped_revision, justification
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

if __name__ == "__main__":
    reference = "Alex looked very handsome in the photo."
    wrong_candidate = "This alteration provides a slightly more formal and polished tone, augmenting the impact of Alex's handsomeness in the photo."
    correct_candidate = "Alex appeared strikingly handsome in the photograph."

    print(f"BLEU score for correct candidate and reference: {calculate_bleu(reference, correct_candidate)}")
    print(f"BLEU score for wrong candidate and reference: {calculate_bleu(reference, wrong_candidate)}")
    original = "Alex looked very handsome in the photo."
    response = """Alex appeared strikingly handsome in the photograph.

Explanation of changes:
1. **Changed ""looked"" to ""appeared""**: This alteration provides a slightly more formal and polished tone, augmenting the impact of Alex's handsomeness in the photo.

2. **Added ""strikingly""**: The inclusion of ""strikingly"" as an adverb intensifies the adjective ""handsome,"" emphasizing the exceptional nature of Alex’s appearance.

3. **Changed ""the photo"" to ""the photograph""**: Substituting ""photo"" with ""photograph"" lends a more formal air to the sentence, aligning with the overall enhanced tone and providing a touch of elegance."""
    print(bleu_split(original, response))