import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from collections import Counter

df = pd.read_csv(r"C:\Users\ANUJ\OneDrive\Desktop\ML-projects\sms_spam_detection\spam.csv", encoding='latin1')
# print(df.head())

# print(df.shape)

#1. Data Cleaning
#2. EDA
#3. Text preprocessing
#4. Model Building
#5. Evaluation
#6. Improvement
#7. Website
#8. Deploy

# print(df.info())

df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

# print(df.sample(5))

#renaming the columns

df.rename(columns={'v1' : 'target', 'v2' : 'text'}, inplace=True)
# print(df.sample(5))

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df['target'] = encoder.fit_transform(df['target'])

# print(df.head())

#missing values
# print(df.isnull().sum())

#check for duplicate values
# print(df.duplicated().sum())

#remove duplicates
df = df.drop_duplicates(keep='first')
# print(df.duplicated().sum())

# print(df.shape)

#EDA

print(df['target'].value_counts())

plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct = "%0.2f")
# plt.show()

#Data is imbalanced

nltk.download('punkt')

#num of characters
df['num_characters'] = df['text'].apply(len)
# print(df.head())

#num of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
# print(df.head())

#num of sentences
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
# print(df.head())

# print(df[['num_characters', 'num_words', 'num_sentences']].describe())

# #ham
# print(df[df['target'] == 0][['num_characters', 'num_words', 'num_sentences']].describe())

#spam
# print(df[df['target'] == 1][['num_characters', 'num_words', 'num_sentences']].describe())

plt.figure(figsize=(10,5))
sns.histplot(df[df['target']==0]['num_characters'], color='blue', label='Ham')
sns.histplot(df[df['target']==1]['num_characters'], color='red', label='Spam')
plt.legend()
# plt.show()

plt.figure(figsize=(10,5))
sns.histplot(df[df['target']==0]['num_words'], color='blue', label='Ham')
sns.histplot(df[df['target']==1]['num_words'], color='red', label='Spam')
plt.legend()
# plt.show()

sns.pairplot(df, hue='target')
# plt.show()

#DATA PRE PROCESSING
# 1. Lower case
# 2.Tokenization
# 3.Removing special characters
# 4.Removing stopwords & punctuations
# 5.Stemming

def transform_text(text) :
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]       
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]       
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)


df['transformed_text'] = df['text'].apply(transform_text)

#Creating Wordcloud for spam

from wordcloud import WordCloud
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))

plt.figure(figsize=(15, 6))
plt.imshow(spam_wc, interpolation='bilinear')  # improves display
plt.axis('off')  # hides axes
# plt.show()

ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))

plt.figure(figsize=(15, 6))
plt.imshow(ham_wc, interpolation='bilinear')  # improves display
plt.axis('off')  # hides axes
# plt.show()

spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
        
# Get top 30 most common words
top_words = pd.DataFrame(Counter(spam_corpus).most_common(30))
top_words.columns = ['word', 'count']  # name the columns

# Plot
plt.figure(figsize=(12,6))
sns.barplot(x='word', y='count', data=top_words)  # use x= and y=
plt.xticks(rotation=90)  # rotate x-axis labels vertically
plt.title("Top 30 words in spam messages")
# plt.show()

ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)
        
# Get top 30 most common words
top_words = pd.DataFrame(Counter(ham_corpus).most_common(30))
top_words.columns = ['word', 'count']  # name the columns

# Plot
plt.figure(figsize=(12,6))
sns.barplot(x='word', y='count', data=top_words)  # use x= and y=
plt.xticks(rotation=90)  # rotate x-axis labels vertically
plt.title("Top 30 words in ham messages")
# plt.show()

#MODEL BUILDING

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df['transformed_text']).toarray()
# print(X.shape)

y=df['target'].values
# print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1)) 

mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))     #Precision:1.0(for 'tfidf')

bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))    

#tfidf ---> MNB

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

svc = SVC(kernel = 'sigmoid', gamma = 1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)

clfs = {
    'SVC': svc,
    'KN': knc,
    'NB': mnb,
    'DT': dtc,
    'LR': lrc,
    'RF': rfc,
    'Adaboost': abc,
    'BgC': bc,
    'ETC': etc,
    'GBDT': gbdt,
    'XGB': xgb
}

from sklearn.metrics import accuracy_score, precision_score

def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    return accuracy, precision


accuracy_scores = []
precision_scores = []

for name, clf in clfs.items():
    current_accuracy, current_precision = train_classifier(clf, X_train, y_train, X_test, y_test)
    
    # print("For", name)
    # print("Accuracy - ", current_accuracy)
    # print("Precision - ", current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
    

performance_df = pd.DataFrame({
    'Algorithm': clfs.keys(),
    'Accuracy': accuracy_scores,
    'Precision': precision_scores
})

temp_df = pd.DataFrame({
    'Algorithm': clfs.keys(), 
    'Accuracy_max_ft_3000': accuracy_scores, 
    'Precision_max_ft_3000':precision_scores
})

# print(performance_df.merge(temp_df, on='Algorithm'))

#model improve


import pickle
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))