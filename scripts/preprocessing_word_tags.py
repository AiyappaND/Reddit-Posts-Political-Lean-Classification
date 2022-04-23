import csv
import re
import string
import nltk
import pathlib
import os
import getopt, sys

import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

def preprocess(input_file, output_file):

    reader = csv.DictReader(input_file)

    wl = WordNetLemmatizer()

    newText = []
    stop_words = set(stopwords.words('english'))
    punc = string.punctuation

    possible_tags = []
    tags_input = []


    sub_reddit = []
    labels_input = []
    score_input = []
    comments_input = []

    skip_tags = ['``',':', '.', "''", '$']
    try:
        for row in reader:
            input_tags = {}
            if row['Political Lean'] == "Liberal":
                labels_input.append(0)
            else:
                labels_input.append(1)

            sub_reddit.append(row['Subreddit'])
            score_input.append(row['Score'])
            comments_input.append(row['Num of Comments'])

            tempText = str.lower(row['Title'] + " " + row['Text'])
            tokenized = word_tokenize(tempText)
            stop_words_removed = []
            for token in tokenized:
                if (token not in stop_words) & (not re.fullmatch('^[' + punc + ']$', token)):
                    stop_words_removed.append(token)

            tempText = ' '.join(stop_words_removed)
            word_pos_tags = nltk.pos_tag(word_tokenize(tempText))
            for word in word_pos_tags:
                tag = "WTAG_" + word[1]
                if word[1] in skip_tags:
                    continue
                if input_tags.get(tag) is None:
                    input_tags[tag] = 1
                else:
                    input_tags[tag] += 1
                if tag not in possible_tags:
                    possible_tags.append(tag)
            tags_input.append(input_tags)

        headers = possible_tags.copy()
        headers.append('subreddit')
        headers.append('score')
        headers.append('num_comments')
        headers.append('label')

        combinedTB_Writer = csv.DictWriter(output_file, headers)
        combinedTB_Writer.writeheader()

        le = LabelEncoder()
        sub_redd_input = le.fit_transform(np.array(sub_reddit))

        for id, tg_input in enumerate(tags_input):
            preprocessed = {}
            for header in headers:
                if re.match('^WTAG_.*', header):
                    if tg_input.get(header) is None:
                        preprocessed[header] = 0
                    else:
                        preprocessed[header] = tg_input[header]
                elif header == 'label':
                    preprocessed[header] = labels_input[id]
                elif header == 'subreddit':
                    preprocessed[header] = sub_redd_input[id]
                elif header == 'score':
                    preprocessed[header] = score_input[id]
                elif header == 'num_comments':
                    preprocessed[header] = comments_input[id]
            combinedTB_Writer.writerow(preprocessed)

        combinedTB.close()
        return newText

    except Exception as e:
        print(e)


def read_from_file(read_file):
    reader = csv.DictReader(read_file)
    newText = []
    try:
        for row in reader:
            newText.append(row['cleaned'])
    except UnicodeDecodeError as e:
        print(row)
        print(e)
    return newText


project_root = pathlib.Path().resolve().parent
inputF = open(os.path.join(project_root, 'data', 'file_name.csv'), "r", encoding="utf8")

generated_dir = os.path.join(project_root, 'data', 'generated')
pathlib.Path(generated_dir).mkdir(parents=True, exist_ok=True)

clean_text = []

try:
    opts, args = getopt.getopt(sys.argv[1:], [], ["no-preprocess"])
except getopt.GetoptError as err:
    # print help information and exit:
    print(err)  # will print something like "option -a not recognized"

no_preprocess = False
for opt, val in opts:
    if opt in ("--no-preprocess"):
        no_preprocess = True

combinedTB = open(os.path.join(generated_dir, 'tokenized_features.csv'), "w+",
                  encoding="utf8", newline='')
preprocess(inputF, combinedTB)


