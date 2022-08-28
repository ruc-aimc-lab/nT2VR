from creat_noun_verb  import *
from posting  import *
from creat_negationdata import *
from checkneg import *
from negscope import *
import argparse
import os
def parse_args():
    parser = argparse.ArgumentParser('prepare')
    parser.add_argument('--root_path', type=str, default="/data1/wzy/VisualSearch/",
                        help='path to datasets.')
    parser.add_argument('--dataset', type=str,default="msrvtt1kAtest",
                        help='dataset collection')
    parser.add_argument('--caption_file', type=str,default="/home/wzy/VisualSearch/msrvtt1kAtest/TextData/msrvtt1k_all.caption.txt",
                        help='origin path of caption')
    parser.add_argument('--test', type=str,default=False,
                        help='origin path of caption')
    parser.add_argument('--cache_dir', type=str, default="/data1/wzy/negbert/",
                        help='path to cache_dir.')
    args = parser.parse_args()
    return args

nlp = spacy.load('en_core_web_sm')
pat = re.compile(
    r"""<\|m\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
    re.IGNORECASE)
nlp.tokenizer = Tokenizer(nlp.vocab, token_match=pat.match)


if __name__ == '__main__':
    opt = parse_args()
    root_path=opt.root_path
    caption_file=opt.caption_file
    dataset=opt.dataset
    cache_dir=opt.cache_dir

    #add negation clue
    stopwords=open("../stopwords_en.txt").read().split("\n")
    stopwords=set(stopwords)
    stopwords2 ={"show","video","clip","person","someone","people","guy","explain","talk","say","speak","argue"}
    stopwords=stopwords&stopwords2
    negfile = os.path.join(root_path, dataset , "TextData" ,dataset + ".caption.negated.new.txt")
    print("start insert negation cue")
    creat_pre_negatindata(dataset,root_path,stopwords,nlp,negfile)

    #delete sentence that maybe have no GT
    #if opt.test:
    #build index for video
    negationf = os.path.join(root_path, dataset, "TextData", dataset + ".caption.negationset.txt")
    stopwords = open("../stopwords_en.txt").read().split("\n")
    stopwords = set(stopwords)
    stopwords2={"show","video","clip","person","someone","people","guy","-","isn't",'go'}
    stopwords=stopwords&stopwords2
    print("start indexing")
    creat_concept(dataset,root_path,caption_file,negationf,stopwords)
    reversed_index=creat_post(dataset,root_path)
    #detect negation scope
    writefile =os.path.join( root_path , dataset , "TextData" , dataset + ".caption.negationinfo_pre.txt")
    print("start getnegscope")
    getnegscope(negfile, writefile,nlp,cache_dir)
    writefile2 = os.path.join(root_path, dataset , "TextData" , dataset + ".caption.negationinfo.txt")
    negifo=postprocess(writefile, writefile2,nlp)
    f2 = os.path.join(root_path,dataset,"TextData",dataset+".negated.new.txt")
    f3 = open(os.path.join(root_path , dataset, "TextData" , dataset + ".negated_withneginfo.txt"), "w")
    f4 =os.path.join(root_path, dataset, "TextData", dataset + ".caption.negation.new.txt")
    print("start delete unlikely sentence")
    # delete  query that maybe have no GT
    check_match(negfile,writefile2,reversed_index,f2,f3,f4,nlp,stopwords,test=opt.test)
    indexf = os.path.join(root_path, dataset, 'TextData', dataset + ".caption.noun_verb_new.txt")
    os.remove(indexf)
    print("finish")