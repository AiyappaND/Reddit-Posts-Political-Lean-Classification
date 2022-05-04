import csv
import re
import string
import nltk
import pathlib
import os
import getopt, sys

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess(input_file, output_file):

    reader = csv.DictReader(input_file)
    combinedTB_Writer = csv.DictWriter(output_file, ["original", "cleaned"])
    combinedTB_Writer.writeheader()
    wl = WordNetLemmatizer()

    newText = []
    stop_words = set(stopwords.words('english'))
    punc = string.punctuation

    empty_count = 0
    original_empty = []
    empty_row_ind = []
    it = 0

    try:
        for row in reader:
            cleanText = ""
            origText = tempText = str.lower(row['Title'] + " " + row['Text'])
            tokenized = word_tokenize(tempText)
            stop_words_removed = []
            for token in tokenized:
                if (token not in stop_words) & (not re.fullmatch('^[' + punc + ']$', token)):
                    stop_words_removed.append(token)

            tempText = ' '.join(stop_words_removed)
            word_pos_tags = nltk.pos_tag(word_tokenize(tempText))
            cleaned = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in
                       enumerate(word_pos_tags)]
            cleanText = ' '.join(cleaned)
            newText.append(cleanText)
            combinedTB_Writer.writerow({'original': origText, 'cleaned': cleanText})
            if (len(cleanText) == 0):
                empty_count += 1
                original_empty.append(origText)
                empty_row_ind.append(it)
            it += 1

        combinedTB.close()
        return newText

    except UnicodeDecodeError as e:
        print(row)
        print(e)

    print(empty_count)
    print("\n".join(original_empty))
    print(empty_row_ind)

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

featureV = open(os.path.join(generated_dir, 'feature_vectors.txt'), "w+",
                encoding="utf8", newline='')

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

if (no_preprocess):
    combinedTB = open(os.path.join(generated_dir, 'title_body.csv'), "r", encoding="utf8")
    clean_text = read_from_file(combinedTB)
else:
    combinedTB = open(os.path.join(generated_dir, 'title_body.csv'), "w+",
                      encoding="utf8", newline='')
    clean_text = preprocess(inputF, combinedTB)

tfidf_vectorizer = TfidfVectorizer(use_idf=True)
print(len(clean_text))
X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(clean_text)

i = 0
for e in X_train_vectors_tfidf:
    #print(e)
    i += 1
    if i == 1:
        break

print(X_train_vectors_tfidf)


featureV.close()

