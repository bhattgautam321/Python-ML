import numpy
import pandas as pd

chatbot1 = pd.read_csv('./totalquestions-alltopics.csv',encoding='latin-1')
#chatbot1.head()

print(chatbot1.iloc[600, 0])
