import json
import numpy as np
import transformers

def preprocess_squad(input_file):
    with open(input_file) as f:
        data = json.load(f)

    
    input_ids = []
    attention_mask = []
    token_type_ids = []
    start_positions = []
    end_positions = []

    count = 0
    
    for article in data['data']:
        print(count*10)
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            count += 1
            print(count)
            for qa in paragraph['qas']:
                print('loading')
                try:

                    question = qa['question']
                    answer = qa['answers'][0]['text']

                    # Tokenize the input using the BERT tokenizer
                    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
                    context_tokens = tokenizer.tokenize(context)
                    question_tokens = tokenizer.tokenize(question)

                    # Find the start and end indices of the answer within the context
                    start_index = context.find(answer)
                    end_index = start_index + len(answer)

                    # Convert the context tokens and question tokens into input ids
                    input_ids_context = tokenizer.convert_tokens_to_ids(context_tokens)
                    input_ids_question = tokenizer.convert_tokens_to_ids(question_tokens)
                    input_ids = input_ids_context + [tokenizer.sep_token_id] + input_ids_question

                    # Create the attention mask for the input ids
                    attention_mask = [1] * len(input_ids_context) + [1] + [1] * len(input_ids_question)

                    # Create the token type ids for the input ids
                    token_type_ids = [0] * len(input_ids_context) + [1] + [1] * len(input_ids_question)

                    # Calculate the start and end positions of the answer tokens in the input ids
                    start_position = len(input_ids_context[:start_index])
                    end_position = len(input_ids_context[:end_index])

                    input_ids.append(input_ids)
                    attention_mask.append(attention_mask)
                    token_type_ids.append(token_type_ids)
                    start_positions.append(start_position)
                    end_positions.append(end_position)

                except:
                    pass
    return np.array(input_ids), np.array(attention_mask), np.array(token_type_ids), np.array(start_positions), np.array(end_positions)
