import json
import os
import openai
from datetime import datetime

openai.organization = 'org-sDd4QQ3oY8PPgOtLXkBp9iO2'
openai.api_key = os.getenv('OPENAI_API_KEY')
response = openai.Model.list()

engine = "text-davinci-003"
prompt = "You're exhausted, your body yawning for sleep."
config = {
    "max_tokens": 50,
    "temperature": 0.0,
}

# Call API to generate text
completion = openai.Completion.create(
    engine=engine, prompt=prompt, **config
)

print(completion)

# write to file
completion['config'] = config
completion['prompt'] = prompt
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
with open(f'../data/completion/{timestamp}.json', 'w', encoding='utf-8') as f:
    json.dump(completion, f, ensure_ascii=False, indent=4)
