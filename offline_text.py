# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = ("../../Models/models--mistralai--Mistral-7B-Instruct-v0.2/"
              "snapshots/cf47bb3e18fe41a5351bc36eef76e9c900847c89")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, 
                                             torch_dtype=torch.float16).to(device)

prompt = "Which is the largest country in the world?"
 
messages = [{"role": "user", "content": prompt}]
encoded_input = tokenizer.apply_chat_template(messages, return_tensors="pt")
input_ids = encoded_input.to(device)
 
# Run Inference
output = model.generate(input_ids, max_new_tokens=100,
                        do_sample=True, top_p=0.95,
                        top_k=1000, temperature=1.0,
                        pad_token_id=tokenizer.eos_token_id)
 
print(tokenizer.decode(output[0], skip_special_tokens=True))