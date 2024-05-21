from openai import OpenAI
import json

def pretty_print(obj):
    for key in obj.__dict__:
        print(f"{key}: {obj.__dict__[key]}")

# Sets up connection to the API, need to put OPEN_API_KEY as a variable in the terminal before running
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "Revise and justify the following sentence: Bill is a congressperson."},
    ],
    max_tokens=None,
    n=2,
)

#print(json.dumps(json.loads(response.model_dump_json()), indent=4))
pretty_print(response)