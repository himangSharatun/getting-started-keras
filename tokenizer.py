import re
def tokenize(string):
    removedNonChar = re.sub(r'[^a-zA-Z0-9]', " ", string)
    removedSpace = re.sub(r'\s+', " ", removedNonChar)
    return removedSpace.lower().split(' ')

def tokenizeSentence(string):
    sentence = ""
    for word in tokenize(string):
        sentence += word + " "
    return sentence