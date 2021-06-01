# Author: Avinash Kumar

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Importing all the required libraries.

paragraph = 'Scorpions are predatory arachnids of the order Scorpiones. They have eight legs, a pair of grasping pincers and a narrow, segmented tail, often carried in a characteristic forward curve over the back and always ending with a stinger. There are over 2,500 described species. They mainly live in deserts but have adapted to a wide range of environments. Most species give birth to live young, and the female cares for the juveniles while their exoskeletons harden, transporting them on her back. Scorpions primarily prey on insects and other invertebrates, but some species take vertebrates. They use their pincers to restrain and kill prey. Scorpions themselves are preyed on by larger animals. Their venomous sting can be used both for killing prey and for defense. Only about 25 species have venom capable of killing a human. In regions with highly venomous species, human fatalities regularly occur.'

sentence  = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()

# Stemming
for i in range(len(sentence)):
    words = nltk.word_tokenize(sentence[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentence[i] = ' '.join(words)
print(sentence)

# Basically what happens when we use the stopwords is that for an example when we wake a sentence like "Hello my name is Avinash Kumar"

# When we apply the stop words on this what happens is that all the words like 'my','is' and 'this' all these which doesn't have any meaning on their own.

# So all these words will be removed and the stemming is applied.

# What is Lemmatization?

# Basically Lemmatization means for an example we have a words like

# History
# Hist
# Histories

# Upon lemmatization on these 3 words it will be converted to history and replaced ( Which has some meaning)

# But in stemming we had the issues and with the help of lemmatization it solved.