# Reddit-Posts-Political-Lean-Classification

Preprocessing scripts

POS tagger:
python scripts\preprocessing_word_tags.py

The raw data should be in data/file_name.csv. 
Features will be generated in data/generated/tokenized_features.csv

Models

Deep NN:
python scripts\NN.py [tokenized_features.csv/w2vecscale.csv]

Requires feature file name after preprocessing. 
Looks for the file in data/generated/

Logistic regression:
python scripts\logisticregr.py [tokenized_features.csv/w2vecscale.csv]

Requires feature file name after preprocessing. 
Looks for the file in data/generated/

SVM:
python scripts\svm.py [tokenized_features.csv/w2vecscale.csv] [linear/rbf/poly]

Requires feature file name after preprocessing and kernel function. 
Looks for the file in data/generated/

Results consolidation

To run all models for a features file:
python scripts\results.py [tokenized_features.csv/w2vecscale.csv]

Requires feature file name after preprocessing. 
Looks for the file in data/generated/
