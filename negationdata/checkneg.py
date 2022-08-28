import random
import ftfy
import html
import regex as re
import os
import numpy as np
import spacy
from spacy.tokenizer import Tokenizer
from nltk.corpus import wordnet as wn
import json
from creat_noun_verb import detect
#处理查询语句
def search(concept,sid,reversed_index):
    res=[]
    for c in concept:
        if c not in reversed_index.keys():
            return []
    sorted(concept,key=lambda x: reversed_index[x][0])
    #带有AND的查询
    res = []

    if concept[0] not in reversed_index.keys():
        return []
    else:
        for vid in reversed_index[concept[0]][1]:
            if vid!=sid:
                res.append(vid)

    for i in range(len(concept)-1):

        q2=concept[i+1]
        if q2 not in reversed_index.keys():
            return []
        res1=res
        res = []
        res2 = reversed_index[q2][1]
        #print(" ".join(map(str,sorted(set(res1) & set(res2)))))
        i,j=0,0
        len1=len(res1)
        len2 = len(res2)
        while(i< len1 and j<len2):
            if res1[i]<res2[j]:
                i+=1
            elif res2[j]<res1[i]:
                j+=1
            else:

                res.append(res1[i])
                i+=1
                j+=1
        if len(res)==0:
            return res

    return res


def substractset(res1,res2):
    res=[]
    i,j=0,0
    len1=len(res1)
    len2 = len(res2)
    while(i< len1 and j<len2):
        if res1[i]<res2[j]:
            res.append(res1[i])
            i+=1
        elif res2[j]<res1[i]:
            j+=1
        else:

            i+=1
            j += 1
    if i< len1:
        res.extend(res1[i:])
    return res
def checkforpos(capinfo,reversed_index,stopwords,nlp):

    #检查有没有符合的
    ssid=capinfo["id"]
    sents=nlp(capinfo["falsesent"].lower())
    concept=set()

    for i, x in enumerate(sents):
        if x.lemma_ in stopwords or x.text in stopwords:
            continue
        if x.pos_ =="NOUN":
                concept.add(x.text)
        elif x.pos_ in ["VERB"]:
            concept.add(str(x.lemma_))
    if  len(concept)==0:
        return -1
    hit=search(list(concept),ssid,reversed_index)

    if len(hit)==0:
        return -1
    else:
        capinfo["match"]=hit
        #print(hit)
    return capinfo





def checkforneg(capinfo,id2falseconcept,reversed_index,stopwords,nlp,cap_negindex):

    #检查有没有更符合的
    ssid=capinfo["id"]
    sents=nlp(capinfo["falsesent"])
    concept=set()
    if ssid       not  in id2falseconcept:

        return -1
    negword=id2falseconcept[ssid]
    for i, x in enumerate(sents) :
        if x.lemma_ in stopwords or x.text in stopwords or i  in cap_negindex[ssid]:
            continue
        if x.pos_ =="NOUN":
            concept.add(x.text)
        elif x.pos_ in ["VERB"]:
            concept.add(str(x.lemma_))
    if not len(negword) or len(concept)==0:
        return -1
    hit=search(list(concept),ssid,reversed_index)
    neghit=search(list(negword),ssid,reversed_index)
    hit=substractset(hit,neghit)
    if len(hit)==0:
        #print('no',ssid,negword,concept,sents,hit)
        return -1

    return capinfo


def check_match(negfile,falseinfofile,reversed_index,f2,f3,f4,nlp,stopwords,test):
    #neg concept
    if test:
        f2=open(f2,"w")
    else:
        f4=open(f4,"w")
    id2falseconcept={}
    cap_negindex={}
    falseinfo=open(falseinfofile).readlines()
    falseinfodict={}
    for i, line in enumerate(falseinfo):
        info = json.loads(line)
        #cap_id= info["cap_id"]
        cap_id = info["cap_id"]
        falseinfodict[cap_id]=info
        cap=info["negcap"]
        id2falseconcept[cap_id]=detect(cap,stopwords,nlp)
        cap_negindex[cap_id]= np.where(np.array(info["negscope"])!=0)[0]
    lines = open(negfile).readlines()
    for i, line in enumerate(lines):
        if i % 10000 == 0:
            print(i)
        capinfo = json.loads(line)
        cap_id = capinfo["id"]
        cands = []
        if "false" in capinfo.keys():
            for index, falseinfo in enumerate(capinfo["false"]):
                # 生成的是肯定句，postive_mask为1
                if "p" in falseinfo["id"]:
                    cand = checkforpos(falseinfo,  reversed_index,stopwords, nlp)
                else:
                    cand = checkforneg(falseinfo, id2falseconcept, reversed_index,stopwords, nlp,cap_negindex)
                if cand != -1:
                    cands.append(cand)

            if len(cands) > 0:
                capinfo['false'] = cands
            else:
                capinfo.pop("false")
            if test:
                for id2, cand in enumerate(cands):
                    falseinfo=capinfo["false"][id2]
                    if falseinfo["id"] in falseinfodict:
                        falseinfodict[falseinfo["id"]]["cap_id"]=cap_id +"#" +str(id2)
                        json.dump(falseinfodict[falseinfo["id"]], f3)
                    else:
                        info = {"cap_id": cap_id +"#" +str(id2), "caption": falseinfo["falsesent"]}
                        json.dump(info, f3)
                    f3.write("\n")
                    wline = cap_id +"#" +str(id2)+" " + cand["falsesent"] + "\n"
                    f2.write(wline)
            else:
                json.dump(capinfo,f4)
                f4.write("\n")
    os.remove(negfile)
    os.remove(falseinfofile)

if __name__ == '__main__':
    dataset="msrvtt1kAtest"
    rootpath="/data4/wzy/VisualSearch/"
    negfile = rootpath+dataset+"/TextData/"+dataset+".caption.mask.txt"

    record = open(rootpath+dataset+"/TextData/"+dataset+".caption.noun_verb_index.txt").readlines()
    falseinfo= rootpath+dataset+"/TextData/"+dataset+".caption.negationinfo.txt"

    stopwords=open("../stopwords_en.txt").read().split("\n")
    nlp = spacy.load('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')
    pat = re.compile(
        r"""<\|m\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
        re.IGNORECASE)
    nlp.tokenizer = Tokenizer(nlp.vocab, token_match=pat.match)

    stopwords=set(stopwords)
    stopwords2=["show","explain","talk","say","speak","argue","video","clip","person","someone","people","guy"]
    for s in stopwords2:
        stopwords.add(s)

    reversed_index = {}
    # 加载倒排索引
    for line in record:
        info = line.strip().split('\t')
        word = info[0]
        freq=info[1]
        docs = info[2].split(' ')
        reversed_index[word] = [freq,docs]

    f2 = open(rootpath+dataset+"/TextData/"+dataset+".caption.falseset.txt", "w")
    f3 = open(rootpath+dataset+"/TextData/"+dataset+".caption.falseset2.txt", "w")
    check_match(negfile, falseinfo, reversed_index, f2, f3,nlp,stopwords)



