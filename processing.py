import os
from tqdm import tqdm
import jsonlines

import outrageclf as oclf
from outrageclf.preprocessing import WordEmbed, get_lemmatize_hashtag
from outrageclf.classifier import _load_crockett_model
from outrageclf.classifier import pretrained_model_predict

input_file = '/afs/crc.nd.edu/user/y/yding4/UIC/fbcomments_6.csv'
output_file = '/afs/crc.nd.edu/user/y/yding4/UIC/fbcomments_6_outrage.jsonl'

instance_list = []

commenter_id = ""
comment_text = ""

# 1. load instance_list
with open(input_file) as reader:
    for line_index, line in enumerate(reader):
        if line_index == 0:
            continue

        elif line == '\n':
            if commenter_id != '' and comment_text != '':
                instance = {
                    'commenter_id': commenter_id,
                    'comment_text': comment_text,
                }
                instance_list.append(instance)

            commenter_id = ""
            comment_text = ""

        else:
            if commenter_id == "":
                parts = line.rstrip('\n').split(',')
                commenter_id = parts[0]
                comment_text = ','.join(parts[1:])
            else:
                comment_text += '\n' + line.rstrip('\n')

    if commenter_id != '' and comment_text != '':
        instance = {
            'commenter_id': commenter_id,
            'comment_text': comment_text,
        }
        instance_list.append(instance)

# 2. utilize outrage_classifier to perform classification
# 2-1 preparation

embedding_url = '/afs/crc.nd.edu/user/y/yding4/UIC/doc_model_files/26k_training_data.joblib'
model_url = '/afs/crc.nd.edu/user/y/yding4/UIC/doc_model_files/GRU.h5'

# loading our pre-trained models
word_embed = WordEmbed()
word_embed._get_pretrained_tokenizer(embedding_url)
model = _load_crockett_model(model_url)

new_instance_list = []
for instance_index, instance in enumerate(tqdm(instance_list)):
    text = [instance['comment_text']]
    lemmatized_text = get_lemmatize_hashtag(text)
    embedded_vector = word_embed._get_embedded_vector(lemmatized_text)
    predict = model.predict(embedded_vector)
    predict_list = predict.tolist()
    # print('predict', predict, 'predict_list', predict_list)
    instance['outrage_predict'] = predict_list
    new_instance_list.append(instance)


# print(new_instance_list[-1].keys())
with jsonlines.open(output_file, 'w') as writer:
    writer.write_all(new_instance_list)
