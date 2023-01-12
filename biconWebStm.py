#!/home/rubentsui/anaconda3/bin/python
# coding: utf-8

import regex as re
import sys
#import time, datetime
import gzip, bz2, lzma
from collections import Counter
import torch
import transformers
import itertools
from lemminflect import getAllInflections #, getAllLemmas
from opencc import OpenCC 

openCC = OpenCC('t2s')

model_files = ["bert-base-multilingual-cased", "xlm-roberta-base", "./model_with_co/", "./model_without_co/"]

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

model_file = model_files[2]
if 'model' not in globals():
    model = transformers.BertModel.from_pretrained(model_file)
    tokenizer = transformers.BertTokenizer.from_pretrained(model_file)
    print(f"Finished loading model [{model_file}] ...")
else:
    print(f"Model [{model_file}] already loaded.")
    

def file_open(filepath):
    #Function to allowing opening files based on file extension
    if filepath.endswith('.gz'):
        return gzip.open(filepath, 'rt', encoding='utf8')
    elif filepath.endswith('.bz2'):
        return bz2.open(filepath, 'rt', encoding='utf8')
    elif filepath.endswith('.xz'):
        return lzma.open(filepath, 'rt', encoding='utf8')
    else:
        return open(filepath, 'r', encoding='utf8')


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '<font color="blue">'
   GREEN = '<font color="#36f307">'
   YELLOW = '\033[93m'
   RED = '<font color="red">'
   BOLD = '<b>'
   UNDERLINE = '\033[4m'
   END = '</b></font>'
   CURSOR_UP = '<font color="blue"><b>' #+ "\033[F" #'\033[1;1H'
   

def flatten(l):  # flatten a nested list
    def flatten0(l):
        for i in l:
            if isinstance(i,list):
                yield from flatten0(i)
            else:
                yield i
    return list(flatten0(l))

def getInflections(s):
    '''
    Get all inflections of the lemma s: Verb, Noun or Adjective
    '''
    infl = getAllInflections(s)
    phr = []
    for t in infl.values():
        phr.extend(list(t))
    if not phr:
        return [s]
    else:
        return list(set(phr))    


def mergeDicts(D1, D2):
    '''
    Input example:
        D1 = {'a':2, 'b':3, 'c': 1}
        D2 = {'b':5, 'c': 0, 'd': 7}
    Output:
        D  = {'a':2, 'b': 3+5, 'c': 1+0, 'd': 7}
    '''
    return dict(Counter(D1) + Counter(D2))


def sortTuples(L):
    '''
    L: list of 2-tuples in the form [(1,2), (1,3), (4,3), (3,10), (4,5), (9,2)]
    Sort by 1st number in tuple then by 2nd number, both in ascending order 
    '''
    return None

def buildLexicon(matched_tokens, alignment, e, z):
    '''
    (1, 3), (1,4), (1, 5) becomes {e[1]: {' '.join([z[3], z[4], z[5]]): 1}}  
    '''
    alignment = sorted(alignment)
    L = dict()
    for (i, j) in alignment:
        if e[i] in matched_tokens:
            s = e[i]; t = z[j]
            if s not in L:
                L[s] = [t]
            else:
                L[s].append(t)
    for k in L:
        v = ' '.join(L[k])
        L[k] = {v: 1}
    
    return L

    

def sentAlignHighlight(s, alignment, e, z):
    # s = list of matches (each "match" is a tuple (i, j) where e[i] is mapped to z[j])
    # e = list of tokens
    # z = list of tokens
    sep = ' '
    e_marked = list(e)
    z_marked = list(z)
    for (i, j) in alignment:
        if e[i] in s:
            src = e[i]; tgt = z[j]
            e_marked[i] = f'{color.RED}{color.BOLD}{e[i]}{color.END}'
            z_marked[j] = f'{color.GREEN}{color.BOLD}{z[j]}{color.END}'
            #e_marked[i] = f'{color.CYAN}{color.BOLD}{e[i]}{color.END}'
            #z_marked[j] = f'{color.YELLOW}{color.BOLD}{z[j]}{color.END}'
    
    return sep.join(e_marked), sep.join(z_marked)



def align_word(src, tgt):
    sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
    token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
    sub2word_map_src = []
    for i, word_list in enumerate(token_src):
        sub2word_map_src += [i for x in word_list]
    sub2word_map_tgt = []
    for i, word_list in enumerate(token_tgt):
        sub2word_map_tgt += [i for x in word_list]

    # alignment
    align_layer = 8
    threshold = 1e-3
    model.eval()
    with torch.no_grad():
        out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
        out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

        softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
        softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

        softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    align_words = set()
    for i, j in align_subwords:
        align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )

    return list(align_words)
    

corpora = """
[0] TWP
[1] Patten
[2] UNPC
[3] TWL
[4] NTURegs
[5] FTV
[6] SAT
[7] CIA
[8] NEJM
[9] VOA
[10] NYT
[11] BBC
[12] Quixote
[13] Wiki
""".strip().split("\n")

C = {k: c.split()[-1]+".xz" for k, c in enumerate(corpora)}


def sz(s, c=0, max_matches=50, stats_only=False):
    '''
    s: Chinese search phrase
    '''
    
    corpus = C[c]
    html_text = []
    Lexicon = dict()
    
    # buit search phrase (sp) regexp
    sp = s.split() # split string into a list of tokens by white spaces
    regex_phr = []
    for s0 in sp:
        inflections = getInflections(s0)
        regex_phr.append(re.compile(fr"\b({'|'.join(inflections)})\b"))
    #html_text += f"regex_phr = {regex_phr}<br>"

    #sys.exit(0)
    if not regex_phr:
        html_text.append(f"Sorry, zero matches found for search phrase [{s}].\n")
        return None

    cnt = 0
    raw_cnt = 0
    num_matches = 0
    with file_open(corpus) as fi:
        for line in fi:
            raw_cnt += 1
            if line.strip().count('\t') < 2:
                continue
                #html_text += f"raw_cnt = [{raw_cnt}]; line fewer than 2 tabs<br>"
                #html_text += line.strip() + '<br>'
                #html_text += '-'*80 + '<br>'
            score, en, zh = line.strip().split('\t', maxsplit=2)
            en = en.replace('``', '‘‘').replace("''", '’’')
            e = en.split()
            z = zh.split()
            en_marked, zh_marked = None, None

            MATCHED_ALL = True
            matches_list = []
            for r in regex_phr:
                matches = r.findall(zh)
                if matches:
                    matches_list.extend(matches)
                MATCHED_ALL &= (len(matches)>0)
            matches_list = list(set(matches_list))
            if MATCHED_ALL:
                #print(f"All words matched!")
                cnt += 1
                #alignments = myaligner.get_word_aligns(en, zh)
                #a = alignments[align_method]
                #a = align_word(en, zh)
                a = align_word(zh, en)
                #print(f"alignment = {a}")
                #print(f"matches_list = {matches_list}")
                zh_marked, en_marked = sentAlignHighlight(flatten(matches_list), a, z, e)
                if stats_only:
                    pass
                else:
                    html_text.append(
                        score + "\n<br>\n" +
                        f'<p class="chinese">\n' +
                        zh_marked + "\n<br>\n" +
                        "</p>" +
                        f'<p class="europe">\n' +
                        en_marked + "\n<br>\n"
                        "</p>"
                    )
                L = buildLexicon(flatten(matches_list), a, z, e)
                for k in L:
                    if k in Lexicon:
                        Lexicon[k] = mergeDicts(Lexicon[k], L[k])
                    else:
                        Lexicon[k] = L[k]
                #print(f"Lexicon per match: {L}")
                #print()
                    
            if cnt >= max_matches: break

    summary = f"No. of matches: {cnt}\n<br>\n"
    #print(Lexicon)
    for k1 in Lexicon:
        v1 = Lexicon[k1]
        #html_text += f"[{k1}]\n"
        s = [(k2, v1[k2]) for k2 in sorted(v1, key=v1.get, reverse=True)]
        for k2, v2 in s:
            if k2:
                summary += f"{v2}\t{k2}\n<br>\n"
    html_text.append(summary)
    
    return html_text


def se(s, c=0, max_matches=50, stats_only=False):
    '''
    s: English search phrase
    '''
    corpus = C[c]
    html_text = []
    Lexicon = dict()

    # buit search phrase (sp) regexp
    sp = s.split() # split string into a list of tokens by white spaces
    # regexp    
    regex_phr = None
    if len(sp) == 1: # regexp for single-word search phrase
        inflections = getInflections(s)
        num = len(inflections) 
        if num > 0:
            regex_phr = [re.compile(fr"\b({'|'.join(inflections)})\b", flags=re.IGNORECASE)]
    else:  # multi-word search phrase
        regex_phr = []
        for s0 in sp:
            inflections = getInflections(s0)
            regex_phr.append(re.compile(fr"\b({'|'.join(inflections)})\b", flags=re.IGNORECASE))

    #sys.exit(0)
    
    if not regex_phr:
        html_text.append(f"Sorry, zero matches found for search phrase [{s}].\n<br>\n")
        return None

    cnt = 0
    raw_cnt = 0
    num_matches = 0
    with file_open(corpus) as fi:
        for line in fi:
            raw_cnt += 1
            if line.strip().count('\t') < 2:
                continue
            score, en, zh = line.strip().split('\t', maxsplit=2)
            en = en.replace('``', '‘‘').replace("''", '’’')
            e = en.split()
            z = zh.split()
            en_marked, zh_marked = None, None

            MATCHED_ALL = True
            matches_list = []
            for r in regex_phr:
                matches = r.findall(en)
                matches_list.append(matches)
                MATCHED_ALL &= (len(matches)>0)
            if MATCHED_ALL:
                #print(f"All words matched!")
                cnt += 1
                #alignments = myaligner.get_word_aligns(en, zh)
                #a = alignments[align_method]
                a = align_word(en, zh)
                en_marked, zh_marked = sentAlignHighlight(flatten(matches_list), a, e, z)
                if stats_only:
                    pass
                else:
                    html_text.append(
                        f"{score}\n<br>\n" + '\t' +
                        f'<p class="europe">\n' +
                        f"{en_marked}\n<br>\n" +
                        "</p>\n" + '\t' +
                        f'<p class="chinese">\n' +
                        f"{zh_marked}\n<br>\n" +
                        "</p>\n"
                    )
                L = buildLexicon(flatten(matches_list), a, e, z)
                for k in L:
                    if k in Lexicon:
                        Lexicon[k] = mergeDicts(Lexicon[k], L[k])
                    else:
                        Lexicon[k] = L[k]
                    
            if cnt >= max_matches: break

    summary = f"No. of matches: {cnt}\n<br>\n"
    #print(Lexicon)
    for k1 in Lexicon:
        v1 = Lexicon[k1]
        #html_text += f"[{k1}]<br>"
        s = [(k2, v1[k2]) for k2 in sorted(v1, key=v1.get, reverse=True)]
        for k2, v2 in s:
            if k2:
                summary += f"{v2}\t{k2}\n<br>\n"
    html_text.append(summary)
    return html_text


regex_zh = re.compile(r"[一-龥]")
    
def s(ss, c=0, max_matches=100, stats_only=False):

    actual_search_function = None
    if regex_zh.findall(ss):  # Chinese characters found
        actual_search_function = sz
        if c in [99]:  # ROCLaws has en, zh reversed 
            actual_search_function = se
        print(f"Search by Chinese: actual search function = [{actual_search_function}]")
    else: # Non-Chinese
        actual_search_function = se
        if c in [99]:  # ROCLaws has en, zh reversed 
            actual_search_function = sz

    return actual_search_function(ss, c=c, max_matches=max_matches, stats_only=stats_only)


def tokenIndices(ss, i, j):
    '''
    Given: input indices (i, j) of the string ss (tokens separated by single spaces),
    Return: the list indices of ss.split() that correspond to the substring ss[i:j]    
    '''
    L = ss.split()
    part1 = ss[:i].split()
    part2 = ss[i:j].split()
    part3 = ss[j:].split()
    return len(part1), len(L) - len(part3)  # these are the list indices


def regex_search(ss, c=0, max_matches=100, stats_only=False):

    zhSearch = False
    if regex_zh.findall(ss):  # Chinese characters found
        zhSearch = True
    
    corpus = C[c]
    results = []
    cnt = 0
    with file_open(corpus) as fi:
        for line in fi:
            line = line.strip()
            try:
                score, en, zh = line.strip().split('\t', maxsplit=2)
            except:
                pass
        
            p = re.compile(fr"({ss})")
            enSub, zhSub = en, zh
            #zh = openCC.convert(zh)
            en = en.replace('``', '‘‘').replace("''", '’’')
            enList = en.split()
            zhList = zh.split()
            if zhSearch:
                #print(f"word alignment = {a}")
                if p.findall(zh):
                    cnt += 1
                    a = align_word(openCC.convert(zh), en)
                    #zhSub = p.sub(fr"{color.GREEN}{color.BOLD}\1{color.END}", zh)
                    for m in p.finditer(zh):
                        ii = m.start()
                        jj = m.end()
                        k, q = tokenIndices(zh, ii, jj)
                        for idx in range(k, q):
                            zhList[idx] = f"{color.RED}{color.BOLD}{zhList[idx]}{color.END}"
                    
                            idxT = [e for (z, e) in a if z == idx]  # target indices
                            #print(f"idxT = {idxT}")
                            for iT in idxT:
                                enList[iT] = f"{color.GREEN}{color.BOLD}{enList[iT]}{color.END}"
                    
                    zhSub = ' '.join(zhList)
                    enSub = ' '.join(enList)
                    lineOut = f'{score}<br/><p class="chinese">{zhSub}</p><br/><p class="europe">{enSub}</p>'
                    results.append(lineOut)
            else:
                #a = align_word(en, zh)
                if p.findall(en):
                    cnt += 1
                    a = align_word(en, openCC.convert(zh))
                    #enSub = p.sub(fr"{color.GREEN}{color.BOLD}\1{color.END}", en)
                    for m in p.finditer(en):
                        ii = m.start()
                        jj = m.end()
                        k, q = tokenIndices(en, ii, jj)
                        for idx in range(k, q):
                            enList[idx] = f"{color.RED}{color.BOLD}{enList[idx]}{color.END}"
                            idxT = [z for (e, z) in a if e == idx]  # target indices
                            for iT in idxT:
                                zhList[iT] = f"{color.GREEN}{color.BOLD}{zhList[iT]}{color.END}"
                            
                    zhSub = ' '.join(zhList)
                    enSub = ' '.join(enList)
                    
                    lineOut = f'{score}<br/><p class="europe">{enSub}</p><br/><p class="chinese">{zhSub}</p>'
                    results.append(lineOut)
            

            if cnt > max_matches: break
    return results
 
 #'''
 #EXAMPLES of REGEX RESEARCH
 #(take[sn]|taking|took) .{1,20} for granted
 #'''
    
if __name__ == '__main__':

    print('\n\n'+'='*100)
    print("""Usage:
    Chinese search phrase    
        s('打擊 犯罪', c=0)
    English search phrase    
        s('preemptive strike', c=0, mac_matches=200) 
    Type C (Capital "C") followed by the <Enter> key to see a list of corpora available.
    """)
    for c in C:
        print(f"c={c}: {C[c][:-3]}")

    
