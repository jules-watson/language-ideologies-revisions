# Gendered language reform and LLM revisions

The structure of the code is as follows:

- `context_social_gender`: contains code, data, and plots for the task on measuring the contextual social gender of About Me sentences.
- `data`: contains some files with data used in experiments (in our case, mainly the role nouns are useful).
- `piloting_[NAME OF PILOT EXPERIMENT]` contains the results for one pilot experiment, which corresponds to one prompt type.
    - `just_word_freqs_[GENDER].csv`: justification word frequencies, corresponding to the revisions that start with the a role noun with gender GENDER
    - `[MODEL NAME]`: contains the outputs and revisions for a specific model
- `random_scripts`: some other scripts and files that were created throughout the process of the work, but do not pertain to our final results. Potentially useful files are in `about_me_gendered_nouns_data`, which contain the role nouns in the About Me dataset with suffixes -man, -woman, -person, and -ess.

Following that, the numbered files represent the steps of running a full experiment, including getting the about me data, running the queries and splitting into revision and justification, and analyzing these results.

`split_algos.py` and `test_split_algos.py` represent some splitting algorithms that we tried out (and their test functions).

`common.py` stores some common files, and `constants.py` contain some constants representing the specific experiment to be run.
