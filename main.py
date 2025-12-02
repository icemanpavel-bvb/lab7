from fastapi import FastAPI
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from pydantic import BaseModel

app = FastAPI()
ru_en = pipeline("translation_ru_to_en", model="Helsinki-NLP/opus-mt-ru-en")
gpt2 = GPT2LMHeadModel.from_pretrained("distilgpt2")
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
en_ru = pipeline("translation_en_to_ru", model="Helsinki-NLP/opus-mt-en-ru")


class Item(BaseModel):
    text: str


@app.post("/chat")
def chat(item: Item):
    # Шаг 1: Русский → Английский
    step1 = ru_en(item.text)[0]['translation_text']

    # Шаг 2: GPT-2 генерация ответа
    inputs = tokenizer(step1, return_tensors="pt")
    outputs = gpt2.generate(**inputs, max_length=100)
    step2 = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Шаг 3: Английский → Русский
    step3 = en_ru(step2)[0]['translation_text']

    return {
        "step_1_russian_input": item.text,
        "step_2_english_translation": step1,
        "step_3_gpt2_response": step2,
        "step_4_russian_output": step3
    }
