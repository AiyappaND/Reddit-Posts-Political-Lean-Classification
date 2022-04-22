import csv
import re
import string
import nltk
import pathlib
import os
import getopt, sys

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess(input_file, output_file):
    reader = csv.DictReader(input_file)
    wl = WordNetLemmatizer()

    newText = []
    stop_words = set(stopwords.words('english'))
    punc = string.punctuation


    possible_tags = []
    tags_input = []
    labels_input = []
    skip_tags = ['``',':', '.', "''", '$']
    try:
        for row in reader:
            input_tags = {}
            tempText = str.lower(row['Title'] + " " + row['Text'])
            if row['Political Lean'] == "Liberal":
                labels_input.append(0)
            else:
                labels_input.append(1)
            tokenized = word_tokenize(tempText)
            stop_words_removed = []
            for token in tokenized:
                if (token not in stop_words) & (not re.fullmatch('^[' + punc + ']$', token)):
                    stop_words_removed.append(token)

            tempText = ' '.join(stop_words_removed)
            word_pos_tags = nltk.pos_tag(word_tokenize(tempText))
            for word in word_pos_tags:
                if word[1] in skip_tags:
                    continue
                if input_tags.get(word[1]) is None:
                    input_tags[word[1]] = 1
                else:
                    input_tags[word[1]] += 1
                if word[1] not in possible_tags:
                    possible_tags.append(word[1])
            tags_input.append(input_tags)

        headers = possible_tags.copy()
        headers.append('label')
        combinedTB_Writer = csv.DictWriter(output_file, headers)
        combinedTB_Writer.writeheader()

        for id, tg_input in enumerate(tags_input):
            preprocessed = {}
            for header in headers:
                if header != 'label':
                    if tg_input.get(header) is None:
                        preprocessed[header] = 0
                    else:
                        preprocessed[header] = tg_input[header]
                else:
                    preprocessed[header] = labels_input[id]
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


