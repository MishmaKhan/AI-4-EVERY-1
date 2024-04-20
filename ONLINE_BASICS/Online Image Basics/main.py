from openai import OpenAI
from apikey import apikey
import os
from PIL import Image
import requests
from io import BytesIO

os.environ['OPEN_API_KEY'] = apikey
OpenAI.api_key = apikey 
client = OpenAI()
prompt = "A cute cat jumping over a fence, cartoon, colorful"

print("Generating Image...")
response = client.images.generate(
    model="dall-e-3",
    prompt=prompt,
    size="1024*1024",
    n=1,)
print(response)

image_url = response.data[0].url
image = requests.get(image_url)
image = Image.open(BytesIO(image.content))
image.show()
# save the image to a file
image.save("generated_image.png")
