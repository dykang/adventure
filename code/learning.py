import os,sys,argparse,path
from sys import argv
import numpy as np
from collections import defaultdict,OrderedDict
from nltk.tokenize import word_tokenize
import tensorflow as tf

sys.path.append( os.path.join(os.path.dirname( os.path.abspath(__file__) ), '..') )
from neu.readers import scitail_reader, snli_reader, scitail_reader_old





class Learning():
  def __init__(self,
      model='default', dataset=None,
      model_path=None,data_path=None,
      overgen=True,
      generator=False, rule=False,
      gan=False, parse=False, rule_apply='hypothesis', order='second',
      subsample=False, ratio=1.0,
      reload=False, beam=False):

    self.name = model
    self.dataset = dataset
    self.model_path = model_path
    self.data_path = data_path #+ '/' + dataset
    self.datatypes = ['train','valid','test']
    self.overgen = overgen

    self.reader = None
    if dataset=='snli_1.0': self.reader = snli_reader
    elif dataset=='scitail': self.reader = scitail_reader_old
    elif dataset=='scitail_1.0': self.reader = scitail_reader
    else: print 'Wrong reader',dataset; sys.exit(1)

    # parameters for RuleGAN learning
    self.generator = generator
    self.rule = rule

    self.gan = gan
    self.parse = parse

    self.rule_apply = rule_apply


    self.ratio = ratio

    self.subsample = subsample #False
    print 'SUBSAMPLE:',self.subsample
    self.order = order

    self.reload = reload
    self.beam = beam

    self.sess = None

  ################################################
  # functions for mdoular/adversarial tensorflow training
  ################################################
  def get_model_dir(self):
    #model_dir = '../../model/' + self.dataset +'/'+ self.name
    data_model_dir = self.dataset +'/'+ self.name
    attr_dirs = []
    for attr in self._attrs:
      if hasattr(self, attr):
        attr_dirs.append("%s=%s" % (attr, getattr(self, attr)))
    attr_dir = '/'.join(attr_dirs)
    return data_model_dir,attr_dir #,data_model_dir


  def save(self, global_step=None):
    print("\n [*] Saving checkpoints...")
    model_name = type(self).__name__
    model_dir,attr_dir = self.get_model_dir()

    checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir, attr_dir)
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    self.saver.save(self.sess,
        os.path.join(checkpoint_dir, model_name), global_step=global_step)


  def get_parameter_size(self):
    total_parameters = 0
    variables = []
    for variable in tf.trainable_variables():
      shape = variable.get_shape()
      variables.append(variable.name)
      variable_parameters = 1
      for dim in shape: variable_parameters *= dim.value
      total_parameters += variable_parameters
    #print 'Total size of parameters',total_parameters
    return variables,total_parameters



  def initialize(self, init=True): # ,  model=None, sess=None): #,

    print 'Intializing parameters...'

    model_name = type(self).__name__
    model_dir,attr_dir = self.get_model_dir()

    log_dir = os.path.join( self.log_dir, model_dir,attr_dir)
    if not os.path.exists(log_dir): os.makedirs(log_dir)


#       all_variables = tf.global_variables() # tf.all_variables()
      # is_not_init = self.sess.run([tf.is_variable_initialized(var) for var in all_variables])
      # uninit_variables = [v for (v, f) in zip(all_variables, is_not_init) if not f]
      # print '\tList of uninit variables [%d]: %s'%(len(uninit_variables), uninit_variables)
      # if len(uninit_variables):
        # self.sess.run(tf.variables_initializer(uninit_variables))
      # # variables,size= self.get_parameter_size()
      # print '\tVariables: %d initialized out of %d'%(len(uninit_variables),len(all_variables))


#    self.sess.run(tf.global_variables())

    uninit_vars = [v.name for v in tf.global_variables()
          if v.name.split(':')[0] in set(self.sess.run(tf.report_uninitialized_variables()))]
    #print 'Uninitialized variables %s'%(' '.join(uninit_vars))
    init_op = tf.variables_initializer(
        [v for v in tf.global_variables()
          if v.name.split(':')[0] in set(self.sess.run(tf.report_uninitialized_variables()))])
    self.sess.run(init_op)
    variables,size= self.get_parameter_size()
    #print 'All variables [%d] %s'%(size,variables)

    self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=3)
    self.load(self.checkpoint_dir)

    self.merged = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter(log_dir, graph=self.sess.graph)


  def load(self, checkpoint_dir, checkpoint_step=None):
    model_dir,attr_dir = self.get_model_dir()
    checkpoint_dir = os.path.join(self.checkpoint_dir,model_dir,  attr_dir)
    print " [*] Loading checkpoints:",checkpoint_dir

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #print (ckpt, ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path and self.reload:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      if checkpoint_step:
        for ckpt_n in ckpt.all_model_checkpoint_paths:
          if int(ckpt_n.split('-')[-1]) == checkpoint_step:
            ckpt_name = os.path.basename(ckpt_n)
            print ' [!] Found!!',ckpt_n
            break
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      print " [*] Load SUCCESS: "+ckpt_name
      return True
    else:
      print" [!] Load FAIL..."
      return False






  # functions for ensemble/mdoular/adversarial neural-symbolic learning

  def load_data_to_text(self, dataset=None, data_path=None, parse=False, limit=False):

    if not dataset: dataset = self.dataset
    if not data_path: data_path = self.data_path

    if dataset == 'snli_1.0':
      datasets = ['%s_%s.jsonl'%(dataset,ds)
          for ds in ['train','dev','test']]
      train_data, val_data, test_data, stats = \
          snli_reader.load_data_to_text(data_path, datasets, parse=self.parse, limit=limit)

    elif dataset.startswith('scitail'):
      version = 'v3'
      if dataset == 'scitail':
        prefix = 'CleanQs4th8thSciQDevTestSplit%s'%(version)
      elif dataset == 'scitail++':
        prefix = 'SourceSplitWithNegativeChoices%s'%(version)
      elif dataset == 'scitailQA':
        prefix = 'SourceSplitWithAllChoices%s'%(version)
      elif dataset == 'scitailQAhard':
        prefix = 'lucene_hard' #%(version)
      else:
        print 'wrong prefix',dataset
        sys.exit(1)

      # load entailment dataset
      datasets = ['entailmentData_%s_%s.tsv'%(prefix,ds)
          for ds in ['train','dev','test']]
      train_data, val_data, test_data, stats = \
          scitail_reader.load_data_to_text(data_path.split('_')[0], datasets)
      print 'loading...',datasets

    elif dataset.startswith('negation'):
      print 'reading..negation file...'
      train,_,_,_ = snli_reader.load_data_to_text_neg(data_path, [dataset,[],[]], parse=self.parse, limit=limit)
      return train



    else:
      print ('wrong dataset',dataset);sys.exit(1)

    print '================================='
    print 'Loading data %s from %s..' %(dataset,data_path)
    print 'Number of entailment: %d/%d/%d'%(len(train_data),len(val_data),len(test_data))
    print '================================='
    return train_data, val_data, test_data



  def align_data(self,data):
    """
      aggregate entailment dataset with KB-tuple dataset by questions
    """
    print '================================='
    print 'Aligning entailment and KB-tuples by questions...'
    #import pdb; pdb.set_trace()
    entails, tuples = data
    qa_entail_tuples = OrderedDict()
    for sent, qa, label in entails:
      qa = ' '.join(qa)
      sent = ' '.join(sent)
      if qa in qa_entail_tuples:
        qa_entail_tuples[qa]['entail'].append((sent, label))
      else:
        qa_entail_tuples[qa] = {'entail':[(sent, label)]}
    print 'Number of question in entailment dataset: %d'%(len(qa_entail_tuples))
    print 'Average number of sentences per question: %.2f'%(np.average([len(entail['entail']) for qa,entail in qa_entail_tuples.items()]))

    tuples = tuples.instances
    num_no_match_qa = 0
    num_no_queries = 0
    for instance in tuples:
      if len( instance.answer_tuples[0]) == 0:
        num_no_queries += 1
        continue

      sub_queries = instance.answer_tuples[0] #instance.label]
      qa = ' '.join(word_tokenize(sub_queries[0].context.lower()))
      if qa in qa_entail_tuples:
        qa_entail_tuples[qa]['tuple'] = instance
      else:
        num_no_match_qa += 1
        #print '\t[NOT MATCHED] ',qa
    print 'Not matched questions (%d) and not queries QA (%d)' %(num_no_match_qa,num_no_queries)
    print 'Number of alignment',len(qa_entail_tuples)
    print 'Avg num of KB per QA: %.2f'%(np.average([len(entail['tuple'].background_tuples) for qa,entail in qa_entail_tuples.items() if 'tuple' in entail]))
    print '================================='
    return qa_entail_tuples



  def load_model(self): #,  switches=[True,True,True]):

    print '\nLoading models for Adventure...\n'

    from neu import main as neural
    from config import NeuralConfig,RuleNetConfig




    if self.generator:
      print ' >> Loading generator..',self.generator
      # laoding Gp and Gn from pre-trained S2S models
      if self.generator in ['all', 'positive']:
        neural_gen_pos_config = NeuralConfig(
            model='S2S', dataset=self.dataset, subsample=self.subsample, training = 'adversarial',
            gen_type = 'positive',reload=True, verbose=False)
        self.gen_pos_model = neural.NeuralReasoner(
            neural_gen_pos_config, test_one=False, is_load_data=False)
        self.gen_pos_train,_,_,_ = self.gen_pos_model.models
        self.sess = self.gen_pos_model.sess
        print ' [**] Positive Generator model loaded'
        print


      if self.generator in ['all', 'negative']:
        neural_gen_neg_config = NeuralConfig(
            model='S2S', dataset=self.dataset,subsample=self.subsample, training = 'adversarial',
            gen_type = 'negative',reload=True, verbose=False)
        self.gen_neg_model = neural.NeuralReasoner(
            neural_gen_neg_config, test_one=False, is_load_data=False)
        self.gen_neg_train,_,_,_ = self.gen_neg_model.models
        self.sess = self.gen_pos_model.sess
        print ' [**] Negative Generator model loaded'
        print


      if self.generator in ['all', 'neutral'] and self.dataset == 'snli_1.0':
        neural_gen_neu_config = NeuralConfig(
            model='S2S', dataset=self.dataset,subsample=self.subsample, training = 'adversarial',
            gen_type = 'neutral',reload=True, verbose=False)
        self.gen_neu_model = neural.NeuralReasoner(
            neural_gen_neu_config, test_one=False, is_load_data=False)
        self.gen_neu_train,_,_,_ = self.gen_neu_model.models
        print ' [**] Neutral Generator model loaded'
        print

    print ' >> Loading discriminator..'
    # loading N model from pre-trained DA models
    neural_config = NeuralConfig(
        model='DA', dataset=self.dataset, subsample=self.subsample, reload=True, verbose=False)
    self.disc_model = neural.NeuralReasoner(
        neural_config, is_load_data=True, parse=True) #, test_one=True)
    self.disc_train, self.disc_valid, self.disc_test, _ = self.disc_model.models
    self.disc_model.test()
    print ' [**] Neural model loaded'
    print








    from RuleNet import RuleNet
    rulenet_config = RuleNetConfig()
    self.rule_model = RuleNet(rulenet_config, batch_size=1)


