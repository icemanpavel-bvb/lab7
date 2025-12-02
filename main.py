from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

class Item(BaseModel):
    text: str

app = FastAPI()

translatortoen = pipeline(
    "translation_ru_to_en",
    model="Helsinki-NLP/opus-mt-ru-en"
)

@app.get("/")
def root():
    return {"message": "Translator API is running"}

@app.post("/translate-to-en/")
def translate(item: Item):
    result = translatortoen(item.text)[0]
    return {
        "translated": result['translation_text']
    }
