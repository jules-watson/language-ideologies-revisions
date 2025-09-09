"""
Unit tests for split_algos.py

Author: Raymond Liu
Date: June 2024
"""

import unittest
import split_algos

class SplitUsingSimilarity(unittest.TestCase):

    def test(self):
        original = "Alex has passed many bills while working as a congressperson."
        response = """Revised sentence: During his tenure as a congressperson, Alex successfully passed numerous bills.

Explanation of changes:
1. **Replacing ""has passed"" with ""successfully passed""**: The addition of ""successfully"" emphasizes the achievement, implying effectiveness and positive outcomes, rather than merely the action of passing bills.

2. **Using ""numerous"" instead of ""many""**: ""Numerous"" often conveys a slightly more formal tone than ""many"" and can imply a greater quantity or abundance, enhancing the impression of Alex's productivity and accomplishments.

3. **Introducing ""During his tenure""**: This phrase specifies the period during which Alex passed the bills and adds a professional nuance, suggesting a sustained performance over time rather than scattered individual achievements. 

These modifications collectively enhance the clarity, tone, and impact of the original sentence, providing a more precise and polished description of Alex's accomplishments."""

        revision, justification = split_algos.split_using_similarity(original, response)
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