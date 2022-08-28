import random
import ftfy
import html
import regex as re
import os
import spacy
from nltk.corpus import wordnet as wn
import json
def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def normalize(input_str):
    input_str = whitespace_clean(basic_clean(input_str))
    res = [input_str]
    replacelist = [("don t", "do not"), ("doesn t", "does not"), ("didn t", "did not"), ("isn t", "is not"),
                   ("aren t", "are not"), ("wasn t", "was not"), ("weren t", "were not"),
                   ("won t", "will not"), ("hasn t", "has not"), ("haven t", "have not"), ("can t", "can not"),
                   ("couldn t", "could not"),
                   ("don't", "do not"), ("doesn't", "does not"), ("didn't", "did not"), ("isn't", "is not"),
                   ("aren't", "are not"), ("won't", "will not"), ("hasn't", "has not"), ("haven't", "have not"),
                   ("can't", "can not"), ("couldn't", "could not"),("wasn't", "was not"), ("weren't", "were not")]
    for pairs in replacelist:
        if input_str.find(pairs[0]) != -1:
            input_str2 = re.sub(pairs[0], pairs[1], input_str)
            res = [input_str, input_str2]
            break
    for pairs in replacelist:
        if input_str.find(pairs[1]) != -1:
            input_str2 = re.sub(pairs[1], pairs[0], input_str)
            res = [input_str2, input_str]
            break
    return res

def process_delneg(cands,negsentence):

    for i,word in enumerate(negsentence):
        if word=="but":
            negsentence[i]="and"
    cands.append({"falsesent": " ".join(negsentence)})
    #print(cands)
    return cands

def delneg(input_str,sid,nlp):
    """
    input：列表
    output：flag和去掉negword后的列表
    """
    cands=[]
    tokens = nlp(input_str)
    mask=None
    input_list = [t.text for t in tokens]
    flag = False

    for i, x in enumerate(input_list):
        if x == "no":
            if len(tokens) > i + 1 and (tokens[i + 1].text not in ["audio", "sound","voice"]):
                input_list.pop(i)
                if tokens[i+1 ].pos_ in ["VERB","ADJ","NOUN"] :
                    cands=process_delneg(cands,input_list)
                elif tokens[i + 1].text  in ["longer"]:
                    input_list.pop(i)
                    cands = process_delneg( cands, input_list)
                flag = True
                break
        elif x == "not":
            input_list.pop(i)
            if len(tokens)==i+1:
                break
            maskindex = i + 1
            while maskindex + 1 < len(tokens) and tokens[maskindex].pos_ not in ["VERB", "ADJ", "NOUN"]:
                maskindex += 1
            if tokens[maskindex].pos_ in ["VERB", "ADJ", "NOUN"]:

                if input_list[i - 1] in ['do', 'did', 'does']:

                    if len(tokens) > maskindex + 1 and tokens[i + 1].pos_ == 'VERB':
                        input_list.pop(i - 1)
                        cands = process_delneg( cands, input_list)


                else:

                    cands = process_delneg( cands, input_list)
                flag = True
            break

        elif x == "without":
            if len(tokens) > i + 1 and (tokens[i + 1].text not in ["audio", "sound","voice"]):
                flag = True
                input_list[i] = "with"
                if tokens[i+1].pos_ in ["VERB", "ADJ", "NOUN"]:
                    cands = process_delneg( cands, input_list)
                elif tokens[i+1].pos_ in ["PRON","DET"]:
                    input_list.pop(i+1)
                    cands = process_delneg( cands, input_list)
                break

    return flag, cands




def add_neg(sents,addition_negindex,maskindex,cuepos,neg, cands,negsentence,ssid,stopwords,nlp):

    negsentence=negsentence.strip()
    if sents[maskindex].lemma_ in stopwords:
        return cands
    startindex = addition_negindex
    if neg in [" with "," without "]:

        startindex=addition_negindex + 1

    negsents= nlp(negsentence)
    cueindex=[3]*len(negsents)
    if len(cuepos)==1:
        try :
            cueindex[cuepos[0]]=1
        except:
            print(negsentence,cuepos,cueindex)
    else:
        cueindex[cuepos[0]:cuepos[1]+1] =[2,2]
    cands.append({"falsesent": negsentence,'cuepos':cueindex})
  
    return cands




# 否定动词
def neg_verb(token, index, sents, cands,sid,stopwords,nlp):
    # tag_ lists the fine-grained part of speech.
    # pos_ lists the coarse-grained part of speech.

    # 三单
    if token.lemma_ in stopwords:
        return cands
    if token.tag_ == "VBZ":
        negsentence = str(sents[:index]) + " does not " + sents[index].lemma_ + " " + str(sents[index + 1:])
        cands=add_neg(sents, index ,index, [index,index+1], " does not ",cands,negsentence,sid,stopwords,nlp)
        negsentence = str(sents[:index]) + " doesn't " + sents[index].lemma_ + " " + str(sents[index + 1:])
        cands=add_neg(sents, index ,index, [index]," doesn't ",cands,negsentence,sid,stopwords,nlp)
    # 现在进行时
    elif token.tag_ == "VBG":
        
        negsentence = str(sents[:index]) + " not " + str(sents[index:])

        cands=add_neg(sents, index ,index, [index]," not ",cands,negsentence,sid,stopwords,nlp)
    # 动词非第三人称单数
    elif token.tag_ == "VBP":
        negsentence = str(sents[:index]) + " do not " + str(sents[index:])
        cands=add_neg(sents, index , index, [index,index+1]," do not ",cands,negsentence,sid,stopwords,nlp)
        negsentence = str(sents[:index]) + " don't " + str(sents[index:])
        cands=add_neg(sents, index , index, [index]," don't ",cands,negsentence,sid,stopwords,nlp)
    # 过去分词
    elif token.tag_ == "VBN":
        if str(sents[index - 1]) != "being":
            negsentence = str(sents[:index]) + " not " + str(sents[index:])
            cands=add_neg(sents,index, index,[index-1,index], " not ",cands,negsentence,sid,stopwords,nlp)

    # 动词过去式
    elif token.tag_ == "VBD":

        negsentence = str(sents[:index]) + " did not " + sents[index].lemma_ + " " + str(sents[index + 1:])
        cands=add_neg(sents, index,index, [index,index+1]," did not ",cands,negsentence,sid,stopwords,nlp)
        negsentence = str(sents[:index]) + " didn't " + sents[index].lemma_ + " " + str(sents[index + 1:])
        cands=add_neg(sents,index, index, [index]," didn't ",cands,negsentence,sid,stopwords,nlp)
    return cands


def neg_sentence(input_str, sid,stopwords,nlp,negation_adj=True):
    sents = nlp(input_str)
    
    cands = list()
    # print(input_str)
    # 否定动词
    lencand=0
    for index, token in enumerate(sents):
        mask=None
        if token.pos_ == "VERB"  :
            # 去掉has no sound...
            if( index == len(sents) - 1 or sents[index + 1].text != 'no' ):

                cands =neg_verb(token, index, sents, cands,sid,stopwords,nlp)
        elif index+1< len(sents):

            if token.text in ["is","are"] and index+1!=len(sents) :
                maskindex = index + 1
                while   maskindex+1 < len(sents) and sents[maskindex].pos_ not in [ "NOUN","VERB"]  and  token.tag_ != "JJ" :
                    maskindex += 1
                if token.text=="is" :
                    negsentence = str(sents[:index + 1]) + "n't " + str(sents[index + 1:])
                    cands = add_neg(sents, index + 1,maskindex,[index] ,"n't ", cands, negsentence,sid,stopwords,nlp)
                elif token.text=="are"  :
                        negsentence = str(sents[:index + 1]) + "n't " + str(sents[index + 1:])
                        cands = add_neg(sents, index + 1,maskindex, [index] ,"n't ", cands, negsentence,sid,stopwords,nlp)
            elif token.text == "being" and index+1!=len(sents) :

                negsentence = str(sents[:index ]) + " not " + str(sents[index :])
                cands = add_neg(sents, index + 1,index+1 ,[index-1,index] , " not ", cands, negsentence,sid,stopwords,nlp)

            elif token.text == "with":
                if  index+2< len(sents)  :
                    # with->without 去掉with no sound...
                    if sents[index + 2].text not in ["sound", "voice", "audio"]:
                        maskindex=index+1
                        while   maskindex+1<len(sents) and sents[maskindex].pos_ !="NOUN":
                            maskindex += 1
                            mask=str(sents[maskindex])
                        if mask is None:
                            continue
                        negsentence = str(sents[:index]) + " without " + str(sents[index + 1:])
                        cands = add_neg(sents,index, maskindex, [index]," without ", cands, negsentence,sid,stopwords,nlp)
                else:
                    negsentence = str(sents[:index]) + " without " + str(sents[index + 1:])
                    cands = add_neg(sents, index,index+1, [index], " without ", cands, negsentence,sid,stopwords,nlp)

    return cands

def creat_pre_negatindata(dataset,rootpath,stopwords,nlp,negfile):
    path = rootpath+dataset+"/TextData/"+dataset+".caption.txt"
    lines = open(path).readlines()

    f2 = open(negfile, "w")

    datas=[]
    for i,line in enumerate(lines):
        if i%10000==0:
            print(i)
        sid, input_str = line.strip().split(None, 1)
        input_strs = normalize(input_str)
        flag, cands = delneg(input_strs[-1],sid,nlp)

        if flag == False:
            cands = neg_sentence(input_strs[-1], sid,stopwords,nlp,negation_adj=False)
            newid0 = sid + "Fn"
            #print(input_str)
            #print(cands)
        else:

            newid0 = sid + "Fp"
        data = {"id": sid, "truth": input_str}
        if len(cands )>0:
            data['false']=[]
            for id2, cand in enumerate(cands):
                newid = newid0 + str(id2)
                cand["id"]=newid
                data['false'].append(cand)

        json.dump(data, f2)

        f2.write("\n")
    datas.append(data)
    return datas



if __name__ == '__main__':
    dataset="msrvtt1kAtest"
    rootpath="/data4/wzy/VisualSearch/"
    nlp = spacy.load('en_core_web_sm')
    pat = re.compile(
        r"""<\|m\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
        re.IGNORECASE)
    nlp.tokenizer = Tokenizer(nlp.vocab, token_match=pat.match)

    stopwords=open("../stopwords_en.txt").read().split("\n")
    stopwords=set(stopwords)
    stopwords2 =["show","explain","talk","say","speak","argue","video","clip","person","someone","people","fail"]
    for s in stopwords2:
        stopwords.add(s)

    creat_pre_negatindata(dataset,rootpath,stopwords,nlp)


