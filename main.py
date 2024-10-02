from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from textblob import TextBlob
from parrot import Parrot
import torch
import random

app = FastAPI()

# Initialize the Parrot paraphrasing model
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False)

class TextData(BaseModel):
    text: str
    api_key: str = None  # Optional API key for authentication

def humanize_text(text):
    # Correct grammar
    blob = TextBlob(text)
    corrected_text = str(blob.correct())

    # Paraphrase text
    paraphrased_sentences = []
    for sentence in corrected_text.split('.'):
        sentence = sentence.strip()
        if sentence:
            paraphrases = parrot.augment(input_phrase=sentence)
            if paraphrases:
                paraphrased_sentence = random.choice(paraphrases)[0]
                paraphrased_sentences.append(paraphrased_sentence)
            else:
                paraphrased_sentences.append(sentence)
    return '. '.join(paraphrased_sentences)

@app.post("/api/humanize")
async def humanize(data: TextData):
    if not data.text:
        raise HTTPException(status_code=400, detail="No text provided.")

    # Optional: Implement authentication using api_key
    # if data.api_key != "your-secure-api-key":
    #     raise HTTPException(status_code=403, detail="Invalid API key.")

    humanized_text = humanize_text(data.text)
    return {"humanized_text": humanized_text}
