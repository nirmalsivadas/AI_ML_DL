!pip install bertviz
# Load model and retrieve attention weights

from bertviz import head_view, model_view
from transformers import BertTokenizer, BertModel

model_version = 'bert-base-uncased'
model = BertModel.from_pretrained(model_version, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_version)
sentence_a = "The cat sat on the mat"
sentence_b = "The cat lay on the rug"
inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt')
input_ids = inputs['input_ids']
token_type_ids = inputs['token_type_ids']

# Run the model and make sure attentions are outputted
outputs = model(input_ids, token_type_ids=token_type_ids, output_attentions=True)
attention = outputs.attentions

sentence_b_start = token_type_ids[0].tolist().index(1)
input_id_list = input_ids[0].tolist() # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(input_id_list)

from bertviz import head_view, model_view

# Visualize with head_view and model_view using standard Hugging Face attention
head_view(attention, tokens, sentence_b_start)
model_view(attention, tokens, sentence_b_start)

