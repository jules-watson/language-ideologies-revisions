"""
Test how well the model outputs can be split into revision and justification sections.

Usage: split.py [insert other usage thingys here]
"""

import json
import re

from config import *

def parse_3_5(response, prompt):
    """
    Parse GPT3.5 short responses.
    """
    if prompt == "short":
        """
        Strategy: Split on the first occurence of \n. Before that is revision, after is justification
        """
        pos = response.find('\n')

        if pos != -1:
            return response[:pos].strip(), response[pos+2:].strip()
        else:
            return None
    elif prompt == "long":
        """
        Strategy: Revision is string in between Sentence: and Explanation:
        Justification is after Explanation:
        """    
        rev_pos = response.find("Sentence:")
        just_pos = response.find("Explanation:")
        if rev_pos != -1 and just_pos != -1:
            return response[rev_pos + len("Sentence: "):just_pos].strip(), response[just_pos + len("Explanation: "):].strip()
        else:
            return None
    else:
        raise Exception("Stimulus not found.")
    


def parse_4(response, prompt):
    """
    Parse GPT4 short responses.
    """
    if prompt == "short":
        """
        Strategy: Split on the first occurence of \n. Before that is revision, after is justification
        """
        pos = response.find('\n')
        begin_pos = len("Revised sentence:") if response.find("Revised sentence:") != -1 else 0

        if pos != -1:
            return response[begin_pos:pos].strip(), response[pos+2:].strip()
        else:
            return None
    elif prompt == "long":
        """
        Strategy: Revision is string in between Sentence: and Explanation:
        Justification is after Explanation:
        """    
        rev_pos = response.find("Sentence:")
        just_pos = response.find("Explanation:")
        if rev_pos != -1 and just_pos != -1:
            return response[rev_pos + len("Sentence: "):just_pos].strip(), response[just_pos + len("Explanation: "):].strip()
        else:
            return None
    else:
        raise Exception("Stimulus not found.")
    

def parse_4o(response, prompt):
    """
    Parse GPT4o short responses.
    """
    if prompt == "short":
        """
        Strategy: Split on the first occurence of \n. Before that is revision, after is justification
        """
        pos = response.find('\n')
        begin_pos = len("Revised sentence:") if response.find("Revised sentence:") != -1 else 0

        if pos != -1:
            return response[begin_pos:pos].strip(), response[pos+2:].strip()
        else:
            return None
    elif prompt == "long":
        """
        Strategy: Revision is string in between Sentence: and Explanation:
        Justification is after Explanation:
        """    
        rev_pos = response.find("Sentence:")
        just_pos = response.find("Explanation:")
        if rev_pos != -1 and just_pos != -1:
            return response[rev_pos + len("Sentence: "):just_pos].strip(), response[just_pos + len("Explanation: "):].strip()
        else:
            return None
    else:
        raise Exception("Stimulus not found.")
    

def parse(response, model, prompt):
    if model == "gpt-3.5-turbo":
        return parse_3_5(response, prompt)
    if model == "gpt-4-turbo":
        return parse_4(response, prompt)
    if model == "gpt-4o":
        return parse_4o(response, prompt)


def analyze_results(output_dir_path, splits_dir_path, model):
    with open(f"{output_dir_path}/{model}.json", 'r') as out_f, open(f"{splits_dir_path}/{model}.txt", 'w') as split_f:
        print(f"Analyzing {model}")
        data = json.load(out_f)

        for prompt in prompts:
            split_f.write(f"Prompt: {prompts[prompt]}\n\n")
            revs, justs = [], []

            for stim in stimuli:
                for response in data[prompt][stim]:
                    res = parse(response, model, prompt)
                    if res:
                        revs.append(res[0])
                        justs.append(res[1])

            
            split_f.write(" Revisions:\n")
            for i, r in enumerate(revs):
                split_f.write(f"    {i+1}) {r}\n")
            split_f.write("\n-----\n")
            split_f.write(" Justifications:\n")
            for i, j in enumerate(justs):
                split_f.write(f"    {i+1}) {j}\n")
        
            split_f.write("\n\n------------------\n\n")
                        
        

if __name__=="__main__":
    for m in models:
        analyze_results("test-split/outputs", "test-split/split-attempt", m)
