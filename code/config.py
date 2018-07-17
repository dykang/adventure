
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

project_dir = '/data/ns'

class NeuralConfig(object):
  def __init__(self,model='DA',training='neural', dataset='scitail', query_decomp=False,
      gen_type = False, vocab_size = False, reload=True,subsample=False,verbose=True):
    self.model = model
    self.dataset = dataset
    self.training = training
    self.project_dir = project_dir
    self.data_path = '%s/data/%s'%(self.project_dir,self.dataset)
    self.vocab_path = '%s/model/%s/%s/vocab'%(self.project_dir,self.dataset,self.training) #, '' if not vocab_size else str(vocab_size))
    self.weights_dir = '%s/model/%s/%s'%(self.project_dir,self.dataset, self.training)
    self.embedding_path = '/data/word2vec/glove.840B.300d.w2v.bin'
    self.query_decomp = False #query_decomp
    self.gen_type = gen_type
    self.is_test = True
    self.verbose = verbose #True
    self.reload = reload
    self.debug = False
    self.subsample = subsample
    self.retrofit = False


class RuleNetConfig(object):
  def __init__(self):
    self.init_scale = 0.01


