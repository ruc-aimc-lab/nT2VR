import json
import os
from checkneg import search
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk import Tree
import nltk
import argparse
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from pattern.en import conjugate, lemma, lexeme, PRESENT, SG
from count_coexsit import *
import random
def parse_args():
    parser = argparse.ArgumentParser('prepare')
    parser.add_argument('--root_path', type=str, default="/data1/wzy/VisualSearch/",
                        help='path to datasets.')
    parser.add_argument('--dataset', type=str,default="msrvtt1kAtest",
                        help='dataset collection')
    parser.add_argument('--caption_file', type=str,default="/home/wzy/VisualSearch/msrvtt1kAtest/TextData/msrvtt1k_all.caption.txt",
                        help='origin path of caption')
    args = parser.parse_args()
    return args
lemmatizer = WordNetLemmatizer()
seed=2
random.seed(seed)
def pattern_stopiteration_workaround():
    try:
        print(lexeme('gave'))
    except:
        pass


def sent(subject,verb1str,verb2str,singular,gender):
    format=random.randint(0,5)
    type=1-format%2
    verb1=verb1str.split(" ")[0]
    verb2 = verb2str.split(" ")[0]
    if subject=="cars":
        a=1
    if (format == 0 or format==1) :
        if singular:
            #print(verb1)
            verb1present = conjugate(verb=verb1, tense=PRESENT, number=SG)
            verb1strpresent = verb1present + " " + " ".join(verb1str.split(" ")[1:])
        else:
            verb1strpresent=verb1str

    else:
        presentverb1 = conjugate(verb1, "part")
        verb2ing = conjugate(verb2, "part")
        verb1ing = conjugate(verb1, "part")

        verb1ingstr = verb1ing + " " + " ".join(verb1str.split(" ")[1:])
        verb2ingstr = verb2ing + " " + " ".join(verb2str.split(" ")[1:])
    if singular:
        be = conjugate(verb="be", tense=PRESENT, number=SG)
        dont="doesn't"
    else:
        be = conjugate("be","2sg")
        dont = "don't"
    if  gender is None:
        if format==0:
            negsent=" ".join([subject,dont ,verb2str , "while" , verb1strpresent])
        elif format==1:
            negsent=" ".join([subject , verb1strpresent , "and", dont, verb2str])
        elif format==2:
            negsent=" ".join([subject,"not", verb2ingstr,"while" , verb1ingstr])
        elif format==3:
            negsent=" ".join([subject,verb1ingstr ,"and not" , verb2ingstr])
        elif format==4:
            negsent=" ".join([subject,be,"not", verb2ingstr , "while" , verb1ingstr])
        elif format==5:
            negsent=" ".join([subject,be,verb1ingstr ,"and not",verb2ingstr])

    else:
        if format == 0:
            negsent=" ".join([subject , dont , verb2str , "and" , gender ,verb1strpresent])
        elif format==1:
            negsent=" ".join([subject, verb1strpresent , "and" , gender ,dont, verb2str])
        elif format==2:
            negsent = subject + " not " + verb2ingstr + " and "  + gender + " " + verb1ingstr

        elif format==3:
            negsent=subject + " " +verb1ingstr + " and not " + verb2ingstr
        elif format==4:
            negsent=" ".join([subject,be,"not", verb2ingstr , "and",gender,be , verb1ingstr])
        elif format==5:
            negsent=" ".join([subject,be,verb1ingstr ,"and not",verb2ingstr])

    return negsent,type


def generate_sent(subject,verb1str,verb2str):
    man = {"man", "boy"}
    woman = {"woman", "lady", "girl"}
    singular = 0
    if "people" in subject:
        singular=0
    negsent=[]
    tokens = word_tokenize(subject)
    tagged_tokens = pos_tag(tokens)
    gender=None
    singular=0

    for tag in tagged_tokens :
        word,pos=tag

        if  pos[:2]=='NN':
            if pos =='NN':
                word2 = lemmatizer.lemmatize(word, "n")
                singular=1
                if word2 in woman:
                    gender="she"
                elif word2 in man:
                    gender="he"
                else:
                    gender =None

            elif pos =='NNS':
                gender = "they"

            break


    if "a " == subject[:2]:
        singular=1
    negsent= sent(subject,verb1str,verb2str,singular,gender)

    return negsent

def generate_falsesent(dataset,root_path,wfile):
    wfile=open(wfile,"w")
    f2path=os.path.join(root_path,dataset,"TextData",dataset + ".caption.2verb_index.txt")
    f2 = open(f2path)
    f3path=os.path.join(root_path,dataset,"TextData",dataset + ".caption.subjectverb_index.txt")
    f3 = open(f3path)
    f4path=os.path.join(root_path,dataset,"TextData",dataset + ".caption.verbstr_index.txt")
    f4 = open(f4path)
    f5path=os.path.join(root_path,dataset,"TextData",dataset + ".caption.concept2cap.txt")
    f5 = open(f5path)
    f6path = os.path.join(root_path, dataset, "TextData", dataset + ".caption.nounstr_index.txt")
    f6 = open(f6path)
    f7path = os.path.join(root_path, dataset, "TextData", dataset + ".caption.subjectverb_index2.txt")
    f7 = open(f7path)
    verb2_index=json.load(f2)
    verbstr_index=json.load(f4)
    concept2cap=json.load(f5)
    nounstr_index=json.load(f6)
    subjectverb_index2 = json.load(f7)
    for k,v in concept2cap.items():
        concept2cap[k]=set(v[1])

    for v1,v2count in verb2_index.items():
        verb2_index[v1]=sorted(v2count.keys(),key=lambda x:v2count[x],reverse=True)
    subjectverb_index=json.load(f3)
    for v1,v2count in subjectverb_index.items():
        subjectverb_index[v1]=sorted(v2count.items(),key=lambda x:len(x[1]),reverse=True)
    for v1,v2count in verbstr_index.items():
        verbstr_index[v1]=sorted(v2count.keys(),key=lambda x:v2count[x],reverse=True)

    for v1,v2count in nounstr_index.items():
        nounstr_index[v1]=sorted(v2count.keys(),key=lambda x:v2count[x],reverse=True)
    numnoun=5
    numverbstr=2
    numnounstr=1
    sid=0
    # verb1变化
    for k,(v1, v2s) in enumerate(verb2_index.items()):
        # verb2变化
        for v2index in range(len(v2s)):
            # 否定的信息关键词就一个，信息太少需要去除
            v2 = v2s[v2index]
            negvs = v2.split(" ")
            if len(negvs) == 1 or "instruction" in v2 or "watch" in v2:
                continue


            #主语变化
            for subjectindex in range(min(len(subjectverb_index[v1]),numnoun)):
                subject=subjectverb_index[v1][subjectindex][0]
                neghit = set()



                # hit
                hit = set(subjectverb_index[v1][subjectindex][1])

                if subject in ["person", "someone", "guy"]:
                    hit = set(concept2cap[v1])
                    break

                for negv in negvs:
                    if subject in subjectverb_index2[negv]:
                        neghit = neghit|set(subjectverb_index2[negv][subject])
                for stopsub in ["person", "someone", "guy"]:
                    if stopsub in subjectverb_index2[negv]:
                        neghit = neghit | set(subjectverb_index2[negv][stopsub])
                hitstr = list(hit - neghit)


                if len(hitstr) > 0:
                    #主语字符串变化
                    for substrindex in range(min(len(nounstr_index[subject]), numnounstr)):
                        substr = nounstr_index[subject][substrindex]
                        # verb1字符串变化
                        for v1strindex in range(min(len(verbstr_index[v1]), numverbstr)):
                            verb1str = verbstr_index[v1][v1strindex]
                            #verb2字符串变化
                            for v2strindex in range(min(len(verbstr_index[v2]), numverbstr)):
                                verb2str=verbstr_index[v2][v2strindex]

                                negsent,type = generate_sent(substr,verb1str,verb2str)
                                if negsent is not None:
                                    json.dump({"cap_id":"s"+str(sid),"caption":negsent,"video_ids":hitstr,"negstionfirst":type,"pos_occur":len(hit),"neg_occur":len(neghit)},wfile)
                                    wfile.write("\n")
                                    sid+=1
    os.remove(f2path)
    os.remove(f3path)
    os.remove(f4path)
    os.remove(f5path)
    os.remove(f6path)
    os.remove(f7path)

if __name__ == '__main__':
    #nltk.download('punkt')
    #nltk.download('averaged_perceptron_tagger')
    #nltk.download('wordnet')
    #nltk.download('omw-1.4')

    pattern_stopiteration_workaround()
    opt = parse_args()
    root_path = opt.root_path
    caption_file = opt.caption_file
    dataset = opt.dataset
    transitivity=pd.read_csv("./verb_transitivity.tsv",delimiter="\t")
    stopwords=open("../stopwords_en.txt").read().split("\n")
    stopwords=set(stopwords)
    stopword_subject=["person","someone","guy","people"]
    stopwords2 = ["show", "video", "clip", "explain", "talk", "discuss", "say", "speak", "argue", "display", "wear"]
    for s in stopwords2:
        stopwords.add(s)
    stopword_verb=["go","make","get","do","try","attempt"]
    negationf = os.path.join(root_path, dataset, "TextData", dataset + ".caption.negationset.txt")

    wfile=os.path.join(root_path,dataset,"TextData", dataset + ".composed.new.txt")
    creat_coexist(dataset,root_path,caption_file,negationf,stopwords,stopword_verb,stopword_subject,transitivity)
    generate_falsesent(dataset,root_path,wfile)