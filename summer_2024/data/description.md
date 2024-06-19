# Description of data files

## Names

`1998.txt` and `camilliere_stimuli.csv` contain the raw data used to generate the names used for the experiment; `names_sampled.json` contains the resulting list of names generated.

The script `0_prepare_names.py` randomly generates a list of names and their gender associations, by taking the names from `1998.txt`and their gender associations from `camilliere_stimuli.csv`.
These resulting names appear in `names_sampled.json`.

## Role nouns

`papineau_role_nouns.json` contains the full list of role nouns, along with their various gendered forms.
`papineau_role_nouns_pilot.json` contains a randomly generated subset of 5 role nouns, used for piloting.

## Stimuli format

`stimuli_format.csv` contains a list of sentence formats used for the experiment.