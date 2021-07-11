# NLP Packages
import numpy as np
import pandas as pd
import nltk
from flask import Flask, render_template, request
from nltk import word_tokenize, sent_tokenize, pos_tag #tokenization # Pos Tag
from nltk.stem import WordNetLemmatizer #Lemmatization
from nltk.corpus import wordnet,stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import operator
import warnings

client=[]
clientm=[]
machine=[]
conv=[]


warnings.simplefilter(action = "ignore", category = RuntimeWarning)

app = Flask(__name__)


tfidf = TfidfVectorizer(min_df=2,max_df=0.6,ngram_range=(1,2))
chatbot1 = pd.read_csv('./totalquestions-alltopics.csv',encoding='latin-1')

#chatbot1.head()

#print(type(chatbot1))
def postag(pos):
    if pos.startswith('N'):
        wp = wordnet.NOUN
    elif pos.startswith('V'):
        wp = wordnet.VERB
    elif pos.startswith('R'):
        wp = wordnet.ADV
    elif pos.startswith('J'):
        wp = wordnet.ADJ
    else:
        wp = wordnet.NOUN

    return wp


wnl = WordNetLemmatizer()


def textprocess(doc):
    # Step1 : Converting into lower case
    doc = doc.lower()
    # Step2 : Remove special characters
    doc = re.sub('[^a-z]', ' ', doc)
    # Step3 : pos tagging (parts of speech)
    token = word_tokenize(doc)  # tokenization -get the words
    token_pos = pos_tag(token)
    # step4 : Lemma and remove stopwords
    lemma = [wnl.lemmatize(word, pos=postag(pos)) for word, pos in token_pos]
    clean = ' '.join(lemma)
    return clean

def cosine(a,b):
    moda = np.linalg.norm(a) # magnitude of a
    modb = np.linalg.norm(b) # magnitude of b
    dotprod = np.dot(a,b) # dot product of vector a and vector b
    # a[0], b[0] -> remove shape in it, we don't want vector to have some shape
    # i.e., neither column matrix nor row matrix
    cos = dotprod/(moda*modb)
    return cos

documents = list(chatbot1['Questions'])
#print(chatbot1)

# Step-1 : Text Processing
documents = [textprocess(doc) for doc in documents] # text processing of the all text

X = tfidf.fit_transform(documents).toarray()
#print('INFO: shape of array =',X.shape)
#print('INFO: Features list',tfidf.get_feature_names())
#print('INFO: length of features =',len(tfidf.get_feature_names()))

df = pd.DataFrame(X)
df.tail(19)


def chats(query):
    global chatbot1
    #user_input = request.form["user_input"]
    client.append(query)
    #user_input = query
    print(client)
    # print(query)
    # step-1: text processing
    clean = textprocess(query)
    # step-2: word embedding (count vectorizer)
    b = tfidf.transform([clean]).toarray()  # query in list

    cosvalue = {}
    for i, vector in enumerate(X):
        cos = cosine(vector, b[0])  # b[0] -> remove shape in it

        if cos > 0.4:
            cosvalue.update({i: cos})  # append values in dictionary

    if len(cosvalue):
        sort = sorted(cosvalue.items(), key=operator.itemgetter(1), reverse=True)
        #print(sort)
        ind = [index for index, cosv in sort[:1]]
        #print(ind)
        # print(ind)
        #print(chatbot1.head())
        # print((chatbot1.iloc[ind[0],1]))
        # response = list(chatbot.loc[ind,1])
        # print(chatbot1)
        response = chatbot1.iloc[ind[0],1]
        # print(response)
        # return response
        machine.append(response)
        # print(machine)
        conv = zip(client, machine)
        conv = list(conv)
        print(conv)
        return conv

    else:
        dont = "I dont understand. Please enter a different question"
        machine.append(dont)
        print("this one is clientm =>", clientm)
        conv = zip(clientm, machine)
        print("this one is machine =>", machine)
        conv = list(conv)
        print("this is conversation 'conv'=>", conv)
        return conv
        pass


#print("Start talking with the bot !")
#print(chats(query = input("You: ")))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chatbot')
def chatbot():
    return render_template('index.html')

@app.route('/resp', methods=['POST'])
def butdothis():
    if request.method == 'POST':
        user_input = request.form['user_input']
        #print(user_input)
        return render_template('index.html', conv=chats(user_input))
    else:
        pass
    # query = ''
    # return render_template('index.html',user_input=chats(query))
    # print(results)

if __name__ == '__main__':
    app.run(host='127.10.0.0', port=int('8000'), debug=True)  ##0.0.0.0.,80
    print("Server started successfully")