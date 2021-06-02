# Author: Avinash Kumar

# Importing the required library
import nltk

paragraph = 'Scorpions are predatory arachnids of the order Scorpiones. They have eight legs, a pair of grasping pincers and a narrow, segmented tail, often carried in a characteristic forward curve over the back and always ending with a stinger. There are over 2,500 described species. They mainly live in deserts but have adapted to a wide range of environments. Most species give birth to live young, and the female cares for the juveniles while their exoskeletons harden, transporting them on her back. Scorpions primarily prey on insects and other invertebrates, but some species take vertebrates. They use their pincers to restrain and kill prey. Scorpions themselves are preyed on by larger animals. Their venomous sting can be used both for killing prey and for defense. Only about 25 species have venom capable of killing a human. In regions with highly venomous species, human fatalities regularly occur.'

# Data Preprocessing ( Cleaning the data )

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# from nltk.stem import PorterStemmer
# I am directly using the lemmatization over stemming you can use steeming as well but lemmatization gives you better results.

lem = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
after_lem = []

for i in range(len(sentences)):
    reg = re.sub('[^a-zA-Z]',' ',sentences[i])
    reg = reg.lower()
    reg = reg.split()
    reg = [lem.lemmatize(word) for word in reg if not word in set(stopwords.words('english'))]
    reg = ' '.join(reg)
    after_lem.append(reg)

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
x = tf.fit_transform(after_lem).toarray()
print(x)

# Summary

# Basically what happens when we use the stopwords is that for an example when we wake a sentence like "Hello my name is Avinash Kumar"

# When we apply the stop words on this what happens is that all the words like 'my','is' and 'this' all these which doesn't have any meaning on their own.

# What is Lemmatization?

# Basically Lemmatization means for an example we have a words like

# History
# Hist
# Histories

# Upon lemmatization on these 3 words it will be converted to history and replaced ( Which has some meaning)

# After that we used a for loop

# 1) First we removed all the unnecessary symbols like '!','.',',' and so on cause these have no use to us.

# 2) We have done that using regular expression where we have use the sub function it basically replaces all the other variables apart from a-zA-z with spaces(' ').

# 3) After that we have coverted all the senteces to lower and split them. The reason why we converted them is for example we have a word 'Good' and 'good' both are pf the same meaning if we don't convert them then they would be taken as two separate words.

# 4) Then we removed all the stop words and appended them into a new list.

# 5) To create TF - IDF model we used something called Tfidfvectorizer

# 6) We create a object and fit transform it and Tfidfvectorizer basically creates a matrix of the words.

# How to understand the output

# In the output we see array with set of numbers so what are these and how to understand these

# we have a total of 11 sentences after sent_tokenize()

# So each of this 11 sentences are converted into based on the frequency after the stop words

# So after that we will calculate two variables one is TF and IDF and combination of these two together will give us the required output.

# i.e TF we calculate it using the forumula TF = (N.O of repetation of words in a sentence / N.O words in the sentence)

#  IDF we calculate it using the forumula IDF = log(N.O of sentence / N.O of sentence the word is in)

# Then the combination of TF*IDF would give us the output array.

# Hope you understand.

