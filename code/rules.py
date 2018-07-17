import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os,sys,json,pdb,re
import codecs
import cPickle as pkl
from collections import OrderedDict,Counter
from nltk.tokenize import word_tokenize


import spacy

import en
# python -m spacy download en
nlp = spacy.load('en')

DATA_DIR='/data/'
MODEL_DIR='/data/ns/model/rules/'
PPDB_RULE=MODEL_DIR + 'ppdb.pkl'
SICK_RULE=MODEL_DIR + 'sick.pkl'

rules = {}
verbose = False # True #False



##########################################
# SICK
###########################################

def filter_sick(input_str):
  input_str = input_str.lower()
  input_str = re.sub(r'-[0-9]*', '', input_str)
  input_str = input_str.strip()
  input_str = ' '.join(word_tokenize(input_str))
  input_str = input_str.encode('utf-8')
  return input_str

sick_types = ['entail','neutral','contradict']
if not os.path.isfile(SICK_RULE):
  sick = {}
  for file in ['sick-train', 'sick-test']:
    with codecs.open(os.path.join(DATA_DIR, 'sick','rrr-0.1',file),'rU','utf-8') as fin:
      next(fin)
      for line in fin:
        rid, pidx, label, lhs, rhs, premise, hypothesis = line.split('\t')
        lhs = filter_sick(lhs)
        rhs = filter_sick(rhs)
        if lhs == rhs: continue
        label = float(label)
        if lhs not in sick:
          sick[lhs] = {'entail':[], 'neutral':[], 'contradict':[]}
        if label == 1.0:
          sick[lhs]['entail'].append(rhs)
        elif label == 0.0:
          sick[lhs]['neutral'].append(rhs)
        elif label == -1.0:
          sick[lhs]['contradict'].append(rhs)
        else:
          print 'wrong label in SICK',line
          sys.exit(1)
  with codecs.open(SICK_RULE, 'wb', 'utf-8') as f_out:
    pkl.dump(sick, f_out)
else:
  sick = pkl.load(codecs.open(SICK_RULE, 'rb'))


cntr = dict(Counter(sick_types))
for lhs, rhs_dict in sick.items():
  for label,rhs_list in rhs_dict.items():
    cntr[label] += len(rhs_list)

print '\tSICK rules %d, ditinct lhs %d'%(sum(cntr.values()), len(sick)),cntr



rules['sick'] = sick

##################################################
# PPDB
##################################################

def filter_ppdb(input_str):
  input_str = input_str.lower()
  input_str = re.sub(r'\[[^)]*\]', '', input_str)
  input_str = input_str.replace('-lrb-', '')
  input_str = input_str.replace('-rrb-', '')
  input_str = input_str.strip()
  input_str = ' '.join(word_tokenize(input_str))
  input_str = input_str.encode('utf-8')
  return input_str



ppdb_types = ['lexical'] #,'o2m', 'phrasal', 'noccg']
if not os.path.isfile(PPDB_RULE):
  ctr = 0
  ppdb = {}
  for ppdb_type in ppdb_types:
    print 'reading...',ppdb_type
    for source_file in ['ppdb-1.0-s-%s'%(ppdb_type)]:
      with codecs.open(os.path.join(DATA_DIR,'ppdb',source_file),'rU','utf-8') as f_in:
        for line in f_in:
          POS, sentA, sentB, _, _ = line.split(' ||| ')
          combo = sentA + "___" + sentB
          comboRev = sentB + "___" + sentA
          POS = POS.replace('[','').replace(']','').encode('utf-8') #.split('\\')
          sentA = filter_ppdb(sentA)
          sentB = filter_ppdb(sentB)
          if sentA == '' or sentB == '' or sentA==sentB: continue

          if sentA not in ppdb: ppdb[sentA] = {t:{} for t in ppdb_types}
          pos_dict = ppdb[sentA][ppdb_type]
          if sentB in pos_dict:
            if POS not in pos_dict[sentB]:
              pos_dict[sentB].append(POS)
          else:
            pos_dict[sentB] = [POS]
          ctr += 1
          if ctr % 50000 == 0 and ctr > 0:
              print '[%d] %s\t%s'%(ctr,sentA,sentB)
  with codecs.open(PPDB_RULE, 'wb', 'utf-8') as f_out:
    pkl.dump(ppdb, f_out)
else:
  ppdb = pkl.load(codecs.open(PPDB_RULE, 'rb'))
cntr = dict(Counter(ppdb_types))
for lhs, rhs_dict in ppdb.items():
  for label,rhs_list in rhs_dict.items():
    cntr[label] += len(rhs_list)
print '\tPPDB rules %d, ditinct lhs %d'%(sum(cntr.values()), len(ppdb)),cntr
rules['ppdb'] = ppdb

#import pdb; pdb.set_trace()
##################################################
# WordNet
##################################################
from nltk.corpus import wordnet as wn

def replace_underbar(input_str): return input_str.replace('_',' ')
def replace_space(input_str): return input_str.replace(' ','_')

def wn_up(input_str, pos=None, max_entities=-1, max_lemmas=-1, verbose=False):
  input_str = replace_space(input_str)
  results = []
  for synset in wn.synsets(input_str):
    if pos and synset.pos() != pos:
      continue
    try:
      parent = synset.hypernym_paths()[0][-2]
    except Exception as e:
      return []
    if verbose:
      print '\tPARENT:',parent, parent.min_depth()
    for l in parent.lemmas()[:max_lemmas]:
      if l.name() != input_str:
        results.append(l.name())
  results = results[:max_entities]
  return [replace_underbar(w) for w in results]


def wn_down(input_str, pos=None, max_entities=-1, max_lemmas=-1, verbose=False):
  input_str = replace_space(input_str)
  hyponyms = []
  for synset in wn.synsets(input_str):
    if pos and synset.pos() != pos:
      continue
    for s in synset.hyponyms()[:max_entities]:
      if verbose:
        print '\tHYPER:',s, s.min_depth()
      for l in s.lemmas()[:max_lemmas]:
        if l.name() != input_str:
          hyponyms.append(l.name())
  hyponyms = hyponyms[:max_entities]
  return [replace_underbar(w) for w in hyponyms]


def wn_part(input_str, pos=None, max_entities=-1, max_lemmas=-1, verbose=False):
  input_str = replace_space(input_str)
  meronyms = []
  for synset in wn.synsets(input_str):
    if pos and synset.pos() != pos:
      continue
    #print 'Part_Meronyms:'
    for s in synset.part_meronyms()[:max_entities]:
      if verbose:
        print '\tMERONY:',s, s.min_depth()
      for l in s.lemmas()[:max_lemmas]:
        if l.name() != input_str:
          meronyms.append(l.name())
  meronyms = meronyms[:max_entities]
  return [replace_underbar(w) for w in meronyms]


def wn_move(input_str, pos=None, max_entities=-1, max_lemmas=-1, verbose=False):
  input_str = replace_space(input_str)
  hyponyms = []
  for synset in wn.synsets(input_str):
    if pos and synset.pos() != pos:
      continue
    # go up
    try:
      parent = synset.hypernym_paths()[0][-2]
    except Exception as e:
      return []
    parents = []
    for l in parent.lemmas()[:max_lemmas]:
      parents.append(l.name())
    # go down
    for synset2 in wn.synsets(parents[0]):
      if pos and synset2.pos() != pos:
        continue
      for s in synset2.hyponyms()[:max_entities]:
        if verbose:
          print '\tMOVE:',parent,parent.min_depth(), s, s.min_depth()
        for l in s.lemmas()[:max_lemmas]:
          if l.name() != input_str:
            hyponyms.append(l.name())
  hyponyms = hyponyms[:max_entities]
  return [replace_underbar(w) for w in hyponyms]


def wn_anto(input_str, pos=None, max_entities=-1, max_lemmas=-1, verbose=False):
  input_str = replace_space(input_str)
  antos = []
  for synset in wn.synsets(input_str):
    if pos and synset.pos() != pos:
      continue
    #print 'Antonyms:'
    #print [(l.name(),l.antonyms()) for l in synset.lemmas() if len(l.antonyms())>0]
    antos = [a.name() for l in synset.lemmas()[:max_lemmas] if len(l.antonyms())>0 for a in l.antonyms() ]
  anots = [a for a in antos if a != input_str]
  antos = antos[:max_entities]
  return [replace_underbar(w) for w in antos]


def wn_syn(input_str, pos=None, max_entities=-1, max_lemmas=-1, verbose=False):
  input_str = replace_space(input_str)
  syns = []
  for synset in wn.synsets(input_str):
    if pos and synset.pos() != pos:
      continue
    for l in synset.lemmas()[:max_lemmas]:
      syns.append(l.name().lower())
  syns = list(set(syns))
  syns = [a for a in syns if a != input_str]
  syns = syns[:max_entities]
  return [replace_underbar(w) for w in syns]



if verbose:
  print 'SYN:',wn_syn('car',pos='n',max_entities=2,max_lemmas=2, verbose=False)
  print 'UP:',wn_up('bad weather',pos='n',max_entities=3,max_lemmas=3, verbose=False)
  print 'DOWN:',wn_down('weather',pos='n',max_entities=3,max_lemmas=3, verbose=False)
  print 'PART:',wn_part('tree',pos='n',max_entities=3,max_lemmas=3, verbose=False)
  print 'MOVE:',wn_move('human',pos='n',max_entities=3,max_lemmas=3, verbose=False)
  print 'ANTO:',wn_anto('cool down',pos='n',max_entities=3,max_lemmas=3, verbose=False)


rules['wordnet'] = {
    'up':wn_up,
    'syn':wn_syn,
    #'down':wn_down,
    #'partof': wn_part,
    #'parallel': wn_move,
    'anto': wn_anto
}


#################################################
# Hand: NEGATE / SWAP / DELETE
#################################################


# input: a is good, output: a is NOT good
def hand_negate(input_str, pos=None, max_entities=-1, max_lemmas=-1, verbose=False):
  parsed = nlp(input_str)
  parsed_dic = {p.dep_:p.orth_ for p in parsed}
  if 'neg' in parsed_dic:
    print 'NEG detected',parsed_dic['neg']
    return ' '.join([p.orth_ for p in parsed if p.dep_ != 'neg'])
  else:
    output_str = []
    for p in parsed:
      if p.dep_ == 'ROOT':
        if p.orth_ in ['am','is','was','were','are']:
          output_str.append(p.orth_)
          output_str.append('not')
        else:
          try:
            present = en.verb.present(p.orth_)
            if present != p.orth_:
              output_str.append('did')
            else:
              output_str.append('do')
          except Exception as e:
            print e
            continue
          output_str.append('not')
          output_str.append(p.orth_)
      else:
        output_str.append(p.orth_)
    return ' '.join(output_str)


# # input: a is good, output: a is NOT good
# def hand_swap(input_str, pos=None, max_entities=-1, max_lemmas=-1, verbose=False):
  # return

# # input: a is good, output: a is NOT good
# def hand_delete(input_str, pos=None, max_entities=-1, max_lemmas=-1, verbose=False):
  # return

if verbose:

  input = u"I did not think that it was the complete set"
  print input
  parsed = nlp(input)
  #import pdb; pdb.set_trace()
  print [(p.dep_, p.orth_) for p in parsed]
  print hand_negate(input)
  print

  input = u"I think that it was the complete set"
  print input
  parsed = nlp(input)
  print [(p.dep_, p.orth_) for p in parsed]
  print hand_negate(input)
  print

  input = u"I compared that it was the complete set"
  print input
  parsed = nlp(input)
  print [(p.dep_, p.orth_) for p in parsed]
  print hand_negate(input)
  print

  input = u"I am sure that it was the complete set"
  print input
  parsed = nlp(input)
  print [(p.dep_, p.orth_) for p in parsed]
  print hand_negate(input)





rules['hand'] = {
   'negate':hand_negate,
   #'swap':hand_swap,
   #'delete':hand_delete
}




# # SNLI
# with codecs.open('complexity.jsonl', 'w', 'utf-8') as f_out:
    # for source_file in ['snli_1.0_dev.jsonl', 'snli_1.0_test.jsonl', 'snli_1.0_train.jsonl']:
        # with codecs.open(source_file,'rU','utf-8') as f_in:
            # for line in f_in:
                # unit = json.loads(line)
                # if unit['gold_label'] == 'entailment':
                    # # Sentence1 entails Sentence2 means that 1 is more complex than 2
                    # out = json.dumps({'label': 1, 'example': unit['sentence1']})
                    # f_out.write(out + '\n')
                    # out = json.dumps({'label': 0, 'example': unit['sentence2']})
                    # f_out.write(out + '\n')

# SICK
# for source_file in ['SICK.txt']:
    # with codecs.open(os.path.join(DATA_DIR,'sick',source_file),'rU','utf-8') as f_in:
        # for line in f_in:
            # _, sentA, sentB, _, _, atob, btoa, _, _, _, _, _ = line.split('\t')
            # if atob == "A_entails_B" and btoa == "B_neutral_A":
                # # A is more complex than B
                # out = json.dumps({'label': 1, 'example': sentA})
                # out = json.dumps({'label': 0, 'example': sentB})
            # if btoa == "B_entails_A" and atob == "A_neutral_B":
                # # B is more complex than A
                # out = json.dumps({'label': 1, 'example': sentB})
           #      out = json.dumps({'label': 0, 'example': sentA})


