# Language Ideology Revisions

Plan: Construct a pipeline consisting of the following:
- Constructing prompts
- Feeding the prompts to LLMs
- Splitting the responses into revision and justification
- Computing statistics about the revisions and justifications

# Feeding the prompts to LLM

Overall goal: construct a system to feed the prompts to LLMs, similar to [this](https://github.com/juliawatson/language-ideologies-2024/blob/main/fall_2023_main/part_2_query_gpt.py)

For now: set the stimuli to be a fixed list of a few sentences

Pipeline is as follows:
- Choose a **model** (saved as an enum)
- Set a **config file**, which tests a specific setting for an experiment (i.e. a specific structure for a query); an example is [here](https://github.com/juliawatson/language-ideologies-2024/blob/main/fall_2023_main/analyses/experiment1/role-nouns-full/config.json). There are 2 types of config files: one for role nouns, and one for pronouns. We focus first on role nouns. It should have:
    - Domain ("role noun")
    - The list of role nouns to test on, augmenting from Papineau et al. ("congressperson", etc.). Saved as a json file
    - The list of names to test on, saved as a csv file
    - Sentence structure: for example,
    "{name} is {a/an} {role noun}"
    - Way of asking: for now,
    "Revise the following sentence and justify the changes."
- Load the **stimuli**: the list of prompts to be fed into GPT, with spaces and options for [FORM]s
    - From this: get a dataframe of input sentences
- **Query the model**: take the input sentences, model (we start with GPT 3.5/4.0), and the output path, and get the output from the model
    - For each input sentence; we generate a corresponding row in the output csv, with the output and a lot of metadata. Rows in output csv are as follows:
        - Prompt
        - Finish reason???? What is this
        - Used number of tokens
        - Message: role and content?
        - Overall: see [this page](https://github.com/juliawatson/language-ideologies-2024/blob/main/summer_code/part_2_generate_outputs.py) along with the OpenAI ChatCompletion documentation
    - Make sure to update + check the number of queries/tokens remaining!

# Questions/concerns to address
- How do we run the code? (most important)
- What exactly is the stimuli? How is it formatted, etc.. Relationship between stimuli and config? 
- What type of instruction sentences are we thinking of doing (i.e. how to word the revise and justify part)
