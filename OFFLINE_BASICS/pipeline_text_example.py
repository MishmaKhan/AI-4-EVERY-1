from transformers import pipeline

# classifier = pipeline(task = "text-classification")
# classifier = pipeline(task = "text-classification",model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english")
model_path = ("../../Models/models--distilbert--distilbert-base-uncased-finetuned-sst-2-english/snapshots/714eb0fa89d2f80546fda750413ed43d93601a13")

classifier = pipeline(task = "text-classification", model = model_path)

print(classifier(["you are the best", "Get Lost"]))
