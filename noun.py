import nltk
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import pymongo
import csv


# converting fetch_20newsgroups to csv file
def twenty_newsgroup_to_csv():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

    df = pd.DataFrame([newsgroups_train.data, newsgroups_train.target.tolist()]).T
    df.columns = ['text', 'target']

    targets = pd.DataFrame(newsgroups_train.target_names)
    targets.columns = ['title']

    out = pd.merge(df, targets, left_on='target', right_index=True)
    out['date'] = pd.to_datetime('now')
    out.to_csv('20_newsgroup.csv')


twenty_newsgroup_to_csv()

# Removing special characters in dataset
spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
              "*", "+", ",", "-", ".", "/", ":", ";", "<",
              "=", ">", "?", "@", "[", "\\", "]", "^", "_",
              "`", "{", "|", "}", "~", "–", " ", "*"]
df = pd.read_csv('20_newsgroup.csv')
for char in spec_chars:
    df['title'] = df['title'].str.replace(char, ' ')

# fetching  the words
data = []
with open(r'20_newsgroup.csv') as f:
    for row in csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE):
        data += row

# Extracting the noun and inserting it into DB table
noun_table = []
for sentence in data:
    for word, pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
        if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
            noun_table.append(word)

# retriveing stored data from MongoDB
for document in noun_table:
    print(document)
