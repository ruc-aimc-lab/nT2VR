import regex as re
import os
import spacy
import json
from nltk.corpus import wordnet as wn
from creat_negationdata import delneg,normalize
def detect(input_str,stopwords,nlp):
    tokens = nlp(input_str)
    verb=set()
    for i, x in enumerate(tokens):
        if x.pos_ in ["VERB"]:
            if x.lemma_ not in stopwords:
                verb.add(str(x.lemma_))
        if x.pos_ in ["NOUN","PROPN"]:
            if x.text not in stopwords:
                verb.add(str(x))

    return verb

def creat_concept(dataset,root_path,readpath,negationfpath,stopwords):
    nlp = spacy.load('en_core_web_sm')
    lines = open(readpath).readlines()
    negationf = open(negationfpath,"w")

    f2 = open(root_path+dataset+"/TextData/"+dataset+".caption.noun_verb_new.txt", "w")
    id2verb={}



    for i, line in enumerate(lines):

        if i % 10000 == 0:
            print(i)
        cap_idfull, input_str = line.strip().split(None, 1)
        input_strs = normalize(input_str)
        flag, cands = delneg(input_strs[-1],cap_idfull,nlp)
        if flag:

            if len(cands):
                negationf.write(line)
                input_str=cands[0]["falsesent"]
            # else:
            #     print(input_strs[-1])
        cap_id = cap_idfull.split('#')[0]

        if cap_id not in id2verb:
            id2verb[cap_id]=set()

        verb = detect(input_str,stopwords,nlp)
        id2verb[cap_id]=id2verb[cap_id]|verb

    for vvid,verb in id2verb.items():
        wline=vvid+" "+" ".join(verb)+"\n"
        f2.write(wline)
    os.remove(negationfpath)

if __name__ == '__main__':
    dataset="msrvtt1kAtest"
    nlp = spacy.load('en_core_web_sm')
    pat = re.compile(
        r"""<\|m\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
        re.IGNORECASE)
    nlp.tokenizer = Tokenizer(nlp.vocab, token_match=pat.match)

    root_path="/data/wzy/VisualSearch/"
    creat_concept(dataset,root_path,nlp)