##Load model directly
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

##Use a pipeline as a high-level helper
from transformers import pipeline

##Language detection library ported from Google's language-detection.
from langdetect import DetectorFactory,detect



##Define input text
sample_text = "où est l'arrêt de bus ?"

##Language detection algorithm is non-deterministic. To enforce consistency, the 'langdetect' libraries readme recommends the below.
DetectorFactory.seed = 0

##Detecting language and prepend.
text_lang = detect(sample_text)
sample_text = ">>" + text_lang + "<<"+ sample_text

#Load pre-trained model and tokenizer. "Helsinki-NLP/opus-mt-mul-en" model used is multi-lang to eng
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en") #"Helsinki-NLP/opus-mt-fr-en" for french to eng
model = TFAutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en")

print(tokenizer(sample_text))

##Using HggingFace pipeline
translator_pipe = pipeline("translation", model=model, tokenizer=tokenizer, clean_up_tokenization_spaces = True)
output = translator_pipe(sample_text)
print(output)





# ##Tokenize input text
# input_ids = tokenizer(sample_text, return_tensors="tf",padding=True, truncation=True)

# ##Perform translation
# translated_text_outputs = model.generate(**input_ids)

# print(translated_text_outputs)

# ##Decode translated text
# translated_text_outputs = tokenizer.decode(translated_text_outputs[0], skip_special_tokens=True)
# print("Generated:", translated_text_outputs )

