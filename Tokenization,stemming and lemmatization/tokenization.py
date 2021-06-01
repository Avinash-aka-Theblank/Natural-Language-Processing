import nltk

# Author: Avinash Kumar

# Code to download all the nltk packages.
# A box will pop up please select all and download all the packages.
nltk.download()

paragraph = 'Scorpions are predatory arachnids of the order Scorpiones. They have eight legs, a pair of grasping pincers and a narrow, segmented tail, often carried in a characteristic forward curve over the back and always ending with a stinger. There are over 2,500 described species. They mainly live in deserts but have adapted to a wide range of environments. Most species give birth to live young, and the female cares for the juveniles while their exoskeletons harden, transporting them on her back. Scorpions primarily prey on insects and other invertebrates, but some species take vertebrates. They use their pincers to restrain and kill prey. Scorpions themselves are preyed on by larger animals. Their venomous sting can be used both for killing prey and for defense. Only about 25 species have venom capable of killing a human. In regions with highly venomous species, human fatalities regularly occur.'

# Tokenizing Sentence
sentence = nltk.sent_tokenize(paragraph)

# Tokenizing word
word = nltk.word_tokenize((paragraph))
# To check the length.
print(len(sentence))
print(len(word))

