import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os,sys,json,pdb,re
import codecs
import cPickle as pkl
from collections import OrderedDict,Counter
from nltk.tokenize import word_tokenize

DATA_DIR='/data/'
OUT_FILE='complexity.json'
MODEL_DIR='../../model/rules/'

rules = {}
verbose = True #False


def filter_sick(input_str):
  input_str = input_str.lower()
  input_str = re.sub(r'-[0-9]*', '', input_str)
  input_str = input_str.strip()
  input_str = ' '.join(word_tokenize(input_str))
  input_str = input_str.encode('utf-8')
  return input_str

##################################################
# Hand: NEGATE / SWAP / DELETE
##################################################

rules['hand'] = {
    'negate':hand_negate,
    'swap':hand_swap,
    'delete':hand_delete
}


from nltk.corpus import wordnet as wn

def replace_underbar(input_str): return input_str.replace('_',' ')
def replace_space(input_str): return input_str.replace(' ','_')

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
    antos = [a.name() for l in synset.lemmas() if len(l.antonyms())>0 for a in l.antonyms() ]
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
    'anto': wn_anto
}


