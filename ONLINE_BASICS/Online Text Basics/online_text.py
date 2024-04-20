from apikey import apikey
from openai import OpenAI
import os

os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey

client = OpenAI()
prompt = "which city is the largest in the world?"

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}])

print("Response")
print(response)
print("Message Content:")
print(response.choices[0].message.content)

# apikey='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'

# def main():
#     pass

# if __name__ == '__main__':
#     main()
