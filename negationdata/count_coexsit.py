import regex as re
import os
import spacy
import json
import pandas as pd
from creat_negationdata import delneg,normalize
from nltk.chunk import *
from nltk.chunk.util import *
from nltk.chunk.regexp import *
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk import Tree
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import random
random.seed(2)
lemmatizer = WordNetLemmatizer()
grammar = r"""
  NP: {<DT|JJ|NN.*>*<NN.*>}          # Chunk sequences of DT, JJ, NN
  PP: {<IN|RP><NP>}               # Chunk prepositions followed by NP
  VP: {<VB.*><NP|PP|CLAUSE>*} # Chunk verbs and their arguments
  VP2: {<VB.*><NN.*><TO><VP>}               # Chunk prepositions followed by NP
  CLAUSE: {<NP><VP>}           # Chunk NP, VP
  """
noun_map={"female":"woman","lady":"woman","male":"man","females":"women","ladies":"women","males":"men"}
#转换动词，名词，形容词，副词，数词到wordnet形式
def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    elif tag==('CD'):
        return "CD"
    return None
def Synonym(word,wn_tag):

    res = wn.synsets(lemmatizer.lemmatize(word, pos=wn_tag), pos=wn_tag)
    if len(res):
        syn =res[0].lemma_names()[0]
    else:
        syn=word
    return syn

def Synonym_withs(word,wn_tag):


    res = wn.synsets(word, pos=wn.NOUN)

    if len(res):
        syn =res[0].lemma_names()[0]
    else:
        syn=word
    if wn_tag=="NNS":
        syn=syn+"s"
    return syn


def lemmatize(word,pos):
    word2 = lemmatizer.lemmatize(word, pos)
    if len(word2)==len(word):
        word2 = lemmatizer.lemmatize(word, "n")
        if len(word2) == len(word):
            word2 = lemmatizer.lemmatize(word, "v")
    return word2

def adddict_verbnoun(dictionary,k1,k2,vid):
    res = dictionary.get(k1, {})
    reslist = res.get(k2, set())
    reslist.add(vid)
    res[k2]=reslist
    dictionary[k1]=res
    return dictionary
def adddict(dictionary,k1,k2):
    res = dictionary.get(k1, {})
    count = res.get(k2, 0)
    res[k2] = count + 1
    dictionary[k1] = res
    return dictionary
def detect(unchunked_text,cap_id,verbnounindex,verbstrdict,concept2id,stopwords,stopword_verb,verb,transitivity,nounstrdict,subjectverb_index2,stopword_subject):

    tokens = nltk.word_tokenize(unchunked_text)
    tagged_tokens = nltk.pos_tag(tokens)
    for i,tag in enumerate(tagged_tokens):
        word,pos=tag
        if word=="girl" and pos=="VB":
            tagged_tokens[i]=(word,pos)
    chunking = nltk.RegexpParser(grammar)
    chunked_text = chunking.parse(tagged_tokens)
    #     if pos[:2] in ['NN', 'VB']:
    #         pos = penn_to_wn(pos)
    #         word=lemmatize(word,pos)
    #         if  word not in stopwords:
    #             concept = Synonym(word, pos)
    #             if concept not in concept2id:
    #
    #                 concept2id[concept] = set()
    #             concept2id[concept].add(cap_id)







    subjecttok = []
    subjectconcept = []
    # for subtree in chunked_text.subtrees():
    #     t = subtree
    #     print(t)
    for subtree in chunked_text.subtrees(filter=lambda t: t.label()in ['VP',"CLAUSE",'VP2']):

        if subtree.label()=="CLAUSE" :
            subjecttok=[]
            subjectconcept=[]
            for subtre in subtree.subtrees(filter=lambda t: t.label()=='NP'):

                for word,pos in subtre.leaves():
                    subjecttok.append(word)
                    if  word not in stopwords:
                        #保留单复数
                        word = Synonym_withs(word, pos)
                        subjectconcept.append(word)




                subject=" ".join(subjecttok)
                subjectconcept=" ".join(subjectconcept)

                if len(subjectconcept)>0:
                    if subject not in concept2id:
                        concept2id[subject]=set()
                    concept2id[subject].add(cap_id)

                    nounstrdict = adddict(nounstrdict, subjectconcept, subject)
                break



        elif subtree.label() == "VP2":
            print(subtree)
        elif subtree.label()=="VP":
            tok=[]
            verbkey=[]
            concept=""
            for index,(word, pos) in enumerate(subtree.leaves()):
                pos = penn_to_wn(pos)
                if pos is not None:
                    word = lemmatize(word, pos)


                if (index == 0 and word in stopwords ) or word == "while":
                    break

                if pos is not None and word not in stopwords:
                    concept = Synonym(word, pos)
                    if word not in stopword_verb:
                        verbkey.append(concept)
                tok.append(word)
            if len(set(verbkey)-set(stopword_subject))==0:
                continue

            if len(verbkey) and len(subjectconcept):


                # if len(tok) == 1:
                #     print(chunked_text)
                try:
                    if len(tok)==1 and verbkey[0] in transitivity.index and transitivity.loc[verbkey[0]]["percent_intrans"]<0.7:
                        continue
                except:
                    pass
                vpstr=" ".join(tok)

                for w in verbkey:
                    subjectverb_index2=adddict_verbnoun(subjectverb_index2, w, subjectconcept, cap_id)
                verbkey=" ".join(verbkey)
                verb.add(verbkey)
                verbnounindex = adddict_verbnoun(verbnounindex, verbkey, subjectconcept,cap_id)


                verbstrdict=adddict(verbstrdict,verbkey,vpstr)
                if verbkey not in concept2id:
                    concept2id[verbkey] = set()
                concept2id[verbkey].add(cap_id)

    return verbnounindex,verbstrdict,concept2id,nounstrdict,verb,subjectverb_index2

def creat_coexist(dataset,root_path,readpath,negationf,stopwords,stopword_verb,stopword_subject,transitivity):
    if dataset=="vatex_test1k5" or dataset=="msrvtt1kAtest"  or dataset=="COCO2014karpathy_test"  :
        filter=100
    elif  dataset=="flickr30ktest" :
        filter=15
    else:
        filter=300
    verb2index={}
    verbnounindex={}
    verbstrdict={}
    concept2id={}
    nounstrdict = {}
    subjectverb_index2={}
    nlp = spacy.load('en_core_web_sm')
    lines = open(readpath).readlines()
    negationf = open(negationf,"w")
    transitivity.set_index("verb",inplace=True)

    verb=set()
    #old_capid= lines[0].strip().split("#", 1)
    for i, line in enumerate(lines):

        if i % 1000 == 0:
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

        verb=set()

        verbnounindex,verbstrdict,concept2id ,nounstrdict,verb,subjectverb_index2= detect(input_strs[-1],cap_id,verbnounindex,verbstrdict,concept2id,stopwords,stopword_verb,verb,transitivity,nounstrdict,subjectverb_index2,stopword_subject)
        verb=list(verb)
        for i in range(len(verb) - 1):
            for j in range(i + 1, len(verb)):
                if verb[i].split(" ")[0] == verb[j].split(" ")[0]:
                    continue
                verb2index = adddict(verb2index, verb[i], verb[j])
                verb2index = adddict(verb2index, verb[j], verb[i])

    #对词条，doc id排序
    #合并
    freq_acount={}
    for word,docid in concept2id.items():
        freq_acount[word]=[len(docid),list(sorted(docid))]

    for key,v in verbstrdict.items():
        kkeys=list(v.keys())
        for k in kkeys :
            if  (len(k.split(" "))==1 and v[k]<filter )or k=="hold" or k=="look":
                v.pop(k)
                #print("poped",k, v)
                verbstrdict[key]=v


    for verbkey,verbdict in verbnounindex.items():
        for nounkey, noundict in verbdict.items():
            verbdict[nounkey]=list(noundict)
        verbnounindex[verbkey]=verbdict

    for verbkey,verbdict in subjectverb_index2.items():
        for nounkey, noundict in verbdict.items():
            verbdict[nounkey]=list(noundict)
        subjectverb_index2[verbkey]=verbdict
    verb2index=list(verb2index.items())

    random.shuffle(verb2index)
    verb2index = dict(verb2index)
    f2path = os.path.join(root_path, dataset, "TextData", dataset + ".caption.2verb_index.txt")
    f2 = open(f2path,'w')
    f3path = os.path.join(root_path, dataset, "TextData", dataset + ".caption.subjectverb_index.txt")
    f3 = open(f3path,'w')
    f4path = os.path.join(root_path, dataset, "TextData", dataset + ".caption.verbstr_index.txt")
    f4 = open(f4path,'w')
    f5path = os.path.join(root_path, dataset, "TextData", dataset + ".caption.concept2cap.txt")
    f5 = open(f5path,'w')
    f6path = os.path.join(root_path, dataset, "TextData", dataset + ".caption.nounstr_index.txt")
    f6 = open(f6path,'w')
    f7path = os.path.join(root_path, dataset, "TextData", dataset + ".caption.subjectverb_index2.txt")
    f7 = open(f7path,'w')
    json.dump(verb2index,f2)
    json.dump(verbnounindex, f3)
    json.dump(verbstrdict, f4)
    json.dump(freq_acount, f5)
    json.dump(nounstrdict, f6)
    json.dump(subjectverb_index2, f7)
if __name__ == '__main__':
    dataset="msrvtt10ktest"
    transitivity=pd.read_csv("./verb_transitivity.tsv",delimiter="\t")

    root_path = "/data4/wzy/VisualSearch/"
    stopwords=open("../stopwords_en.txt").read().split("\n")
    stopwords=set(stopwords)
    stopword_subject=["person","someone","guy","people"]
    stopwords2 =["show","video","clip","explain","talk","say","speak","argue","display","wear"]
    for s in stopwords2:
        stopwords.add(s)
    stopword_verb=["go","make","get","do"]
    readpath = root_path + dataset + "/TextData/" + dataset + ".caption.txt"
    #readpath = root_path +  "msrvtt1kAval" + "/TextData/" + "msrvtt1kAval" + ".caption2.txt"
    negationf = os.path.join(root_path, dataset, "TextData", dataset + ".caption.negationset.txt")

    creat_coexist(dataset,root_path,readpath,negationf,stopwords,stopword_verb,transitivity)