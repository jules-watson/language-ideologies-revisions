import step_6_justifications


def test_mask_quoted_strings_simple_double_quotes():
    role_noun_set = ['meteorologist', 'weatherman', 'weatherwoman']
    input_str = '**Changed "your weatherwoman" to "sharing the weather with you"**: The original phrase "your weatherwoman" is a bit informal and might come across as possessive.'
    expected = '**Changed [MASK] to [MASK] **: The original phrase [MASK] is a bit informal and might come across as possessive.'
    actual = step_6_justifications.mask_quoted_strings(input_str, role_noun_set)
    assert actual == expected


def test_mask_quoted_strings_simple_single_quotes():
    role_noun_set = ['meteorologist', 'weatherman', 'weatherwoman']
    input_str = "'Trusted weather forecaster' is a more accurate title than 'weatherwoman'."
    expected = "[MASK] is a more accurate title than [MASK] ."
    actual = step_6_justifications.mask_quoted_strings(input_str, role_noun_set)
    assert actual == expected


def test_mask_quoted_strings_ignores_single_quotes_in_contractions():
    role_noun_set = ['meteorologist', 'weatherman', 'weatherwoman']
    input_str = "'Weatherwoman' is an outdated term, and it's more common to use the term 'meteorologist' for someone who studies weather."
    expected = "[MASK] is an outdated term, and it's more common to use the term [MASK] for someone who studies weather."
    actual = step_6_justifications.mask_quoted_strings(input_str, role_noun_set)
    assert actual == expected


def test_mask_quoted_strings_masks_role_nouns():
    role_noun_set = ['meteorologist', 'weatherman', 'weatherwoman']
    input_str = '**Used more descriptive language**: Instead of saying "something I had always wanted to be," I used "my lifelong passion," which is a more vivid and engaging way to describe the speaker\'s desire to be a meteorologist.'
    expected = '**Used more descriptive language**: Instead of saying [MASK] I used [MASK] which is a more vivid and engaging way to describe the speaker\'s desire to be a [MASK] .'
    actual = step_6_justifications.mask_quoted_strings(input_str, role_noun_set)
    assert actual == expected


def test_mask_quoted_strings_masks_role_nouns_in_quotes():
    # I added this test because there were several cases like this for llama-3.1-8B-Instruct,
    # where a quoted role noun usage appeared without a space before the opening quote
    role_noun_set = ['meteorologist', 'weatherman', 'weatherwoman']
    input_str = "Using'meteorologist' ensures that the language is inclusive and respectful of your friend's identity."
    expected = "Using [MASK] ensures that the language is inclusive and respectful of your friend's identity."
    actual = step_6_justifications.mask_quoted_strings(input_str, role_noun_set)
    assert actual == expected


def test_mask_quoted_strings_role_nouns_capitalized():
    role_noun_set = ['meteorologist', 'weatherman', 'weatherwoman']
    input_str = "**Weatherman to Meteorologist:** While 'weatherman' is a commonly understood term, 'meteorologist' is more professional and specific."
    expected = "** [MASK] to [MASK] :** While [MASK] is a commonly understood term, [MASK] is more professional and specific."
    actual = step_6_justifications.mask_quoted_strings(input_str, role_noun_set)
    assert actual == expected
