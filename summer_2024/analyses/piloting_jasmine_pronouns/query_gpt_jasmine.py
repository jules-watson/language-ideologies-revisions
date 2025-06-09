"""Small script for querying GPT models. 

To run this script: 
1. Install the openai library with pip 
2. In the commandline, run the command: export OPENAI_API_KEY="{API-KEY-HERE}" 
"""

from openai import OpenAI 

N_RESPONSES = 1 
MAX_TOKENS = 500 
GPT_MODEL_NAME = "gpt-4o" # For a list of model names, see: https://platform.openai.com/docs/pricing 

client = OpenAI() 

def query_gpt(prompt): 
    result = client.chat.completions.create( 
        model=GPT_MODEL_NAME, 
        messages=[ 
            {"role": "user", "content": prompt}, 
        ], 
        max_tokens=MAX_TOKENS, 
        n=N_RESPONSES, 
    ) 
    return result.choices[0].message.content 

if __name__ == "__main__": 
    result = query_gpt("Give me a list of vegetable related puns.") 
    print(result)