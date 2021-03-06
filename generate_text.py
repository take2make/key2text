import re
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('trained_model', 
                                                return_dict=True)

# ------------------------------------------------------------------ #
#          generate text after correct preprocessing data            #
# ------------------------------------------------------------------ #

def generate(text):
    """
    generate new text by using text
    input: text - keywords
    output: result - text from keywords
    """
    texts = text.split(".")
    result = ""
    for txt in texts:
        model.eval()
        input_ids = tokenizer.encode("WebNLG:{} </s>".format(txt), 
                                   return_tensors="pt")  
        outputs = model.generate(input_ids)
        result += tokenizer.decode(outputs[0])
    result = re.sub('<pad>|</s>',"",result)
    return result
