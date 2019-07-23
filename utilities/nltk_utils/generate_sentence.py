import sys
from nltk.tokenize import sent_tokenize



#sent_detector = TweetTokenizer() #nltk.data.load('tokenizers/punkt/english.pickle')

with open(sys.argv[1], "r") as f_in, open(sys.argv[2], "w", encoding="utf-8") as f_out:
    for line in f_in:
        text = line.strip()
        stmts = sent_tokenize(text)
        last_stmt = stmts[-1].split(',')[-1]
        f_out.write("{}\t{}\n".format(text, last_stmt))


