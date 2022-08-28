import os
import regex as re
import argparse
def creat_post(dataset,root_path):
    indexf=os.path.join(root_path,dataset,'TextData',dataset+".caption.noun_verb_new.txt")
    reader=open(indexf)
    term_doc=[]
    for line in reader.readlines():
        cap_id, caption = line.strip().split(None, 1)

        #去重
        corpus=set(re.split("\t| |\n",caption))
        term_doc.extend(list(map(lambda x:(x,cap_id),corpus) ))


    #对词条，doc id排序
    term_doc=sorted(term_doc,key=lambda x:(x[0],x[1]))
    #合并
    freq_acount={}
    lastword=""
    for word,docid in term_doc:
        #已初始化情形,需要添加
        if word == lastword :
            freq_acount[word][0]+=1
            freq_acount[word][1].append(docid)
        #未初始化情形
        elif word != lastword:
            docs=list()
            freq_acount[word]=[[],docs]
            freq_acount[word][0]=1
            freq_acount[word][1].append(docid)
            lastword=word

    # f2 = open(root_path+dataset+"/TextData/"+dataset+".caption.noun_verb_index.txt", 'w')
    # for k,account in freq_acount.items():
    #     word =k
    #     count = str(account[0])
    #     docs = account[1]
    #     docs = sorted(list((docs)))
    #     docs = list(map(lambda one: str(one), docs))
    #     docs = " ".join(docs)
    #     line = word + '\t' + count + '\t' + docs
    #     f2.write(line + '\n')
    return freq_acount

if __name__ == '__main__':
    dataset="msrvtt1kAtest"
    root_path="/home/wzy/VisualSearch/"
    creat_post(dataset,root_path)
