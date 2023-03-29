import nltk
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

review = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

review.head(2)

review.info()

review.describe()

review.columns

review['Liked'].nunique()

print(review['Liked'].unique())

review.head()

plt.figure(figsize=(10, 5))
sns.countplot(x=review.Liked)

(x, y) = (review['Review'].values, review['Liked'].values)

nltk.download('stopwords')
corpus = []

for i in range(0, len(review)):
    reviews = re.sub('[^a-zA-Z]', ' ', review['Review'][i])
    reviews = reviews.lower()
    reviews = reviews.split()

    ps = PorterStemmer()
    reviews = [ps.stem(word) for word in reviews if not word in set(
        stopwords.words('english'))]

    reviews = ' '.join(reviews)
    corpus.append(reviews)

corpus

one_hot_encoded_data = pd.get_dummies(review, columns=['Review'])
print(one_hot_encoded_data)
# col = ["Review", "Encode"]
# data = []
# for encode in one_hot_encoded_data:
#   data.append([encode.Review])
# df = pd.DataFrame(data, columns=col)
# print(df)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=0)

x_train.shape

x_test.shape

y_train.shape

y_test.shape

# **Import** **CountVectorizer**

vect = CountVectorizer(stop_words='english')
print(vect)

x_train_vect = vect.fit_transform(x_train)
x_test_vect = vect.transform(x_test)

# x_test_vect
# x_train_vect

# **SVC** **from** **SVM**

model = SVC()

model.fit(x_train_vect, y_train)

y_pred = model.predict(x_test_vect)

accuracy_score(y_pred, y_test)

# **Without ** **Pipeline** 


vect = CountVectorizer()
tfidf = TfidfTransformer()
clf = SGDClassifier()
vX = vect.fit_transform(x_train)
tfidfX = tfidf.fit_transform(vX)
# predicted = clf.fit_predict(tfidfX)

vX = vect.fit_transform(x_test)
tfidfX = tfidf.fit_transform(vX)
# predicted = clf.fit_predict(tfidfX)

# **With ** **Pipeline** ⭐⭐⭐

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])
# predicted = pipeline.fit(x_train).predict(x_train)
# predicted = pipeline.predict(x_test)

text_model = make_pipeline(CountVectorizer(), SVC())

text_model.fit(x_train, y_train)

y_pred = text_model.predict(x_test)

y_pred


# **Evaluate**

accuracy_score(y_pred, y_test)

# **Save the Model**

joblib.dump(text_model, 'Project')

text_model = joblib.load('Majors')

text_model.predict(['hello!!Love Your Food'])

text_model.predict(
    ["omg!!it was too spice and i asked you don't add too much "])

# review.head()
len(review)

review.isnull().sum()

cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = review.iloc[:, 1].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=0)

classifier1 = SVC()
classifier1.fit(x_train, y_train)

classifier2 = MultinomialNB()
classifier2.fit(x_train, y_train)

y_pred1 = classifier1.predict(x_test)
y_pred2 = classifier2.predict(x_test)

print(metrics.classification_report(y_test, y_pred1))
print(metrics.classification_report(y_test, y_pred2))

print("SVC: ", (metrics.accuracy_score(y_test, y_pred1))*100, "%")
print("MultinomialNB: ", (metrics.accuracy_score(y_test, y_pred2))*100, "%")
