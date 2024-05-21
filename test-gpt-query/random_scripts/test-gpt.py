"""
Test the GPT3.5 chat completions API.
"""

from openai import OpenAI

def pretty_print(obj):
    """
    Print the attributes of obj, separated by \n
    """
    for key in obj.__dict__:
        print(f"{key}: {obj.__dict__[key]}")

if __name__ == "__main__":
    # Set up connection to the API, need to put OPEN_API_KEY as a variable in the terminal before running
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Revise and justify the following sentence: Bill is a congressperson."},
        ],
        max_tokens=None,
        n=2,
    )

    # Print the attributes of response
    pretty_print(response)