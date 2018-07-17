from nltk.tree import Tree
import nltk

def is_negate(sent):
  neg_terms = ['not', 'no', 'n\'t', 'don\'t', 'didn\'t', 'doesn\'t', 'never']
  for w in sent:
    if w in neg_terms or w.endswith('n\'t'):
      return True
  return False



def flatten(sent): return ' '.join(str(sent).split())


def transform(sent, lhs, rhs): return sent.lower().replace(lhs,rhs)

def extract_phrase_from_parse(sent_tree):
  #tree = Tree.fromstring(sent)

  phrases = traverseTree(sent_tree)
  return [(pos[0].lower(), ' '.join(phrase).lower() )  for pos, phrase in phrases]

def traverseTree(tree,tags=[]):
  phrases = []
  for subtree in tree:
    if type(subtree) == nltk.tree.Tree:
      if subtree.label().startswith("N"):
        phrases.append((subtree.label(), subtree.leaves()))
      if subtree.label().startswith("V"):
        phrases.append((subtree.label(), subtree.leaves()))
      phrases += traverseTree(subtree)
  return phrases


def write_f(output, verbose=False,fwrite=False):
  if verbose:
    print output
  if fwrite:
    fwrite.write('%s\n'%('\t'.join([str(w) for w in output])))
    fwrite.flush()

def import_embeddings(filename):
  import gensim
  glove_embedding = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
  return glove_embedding



