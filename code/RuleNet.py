"""
Get (rtype, lhs,rhs, g,p,h,r1,r2,r3)
RuleNet (gamma):
  (p+h, lhs+rhs) -> p:0.75 (e.g., use or not)
  c = Encoder_D(p+h) (from D)
  P( p | r,c)
  R = [SICK-12511,PPDB-23273,WN-3,HAND-2]
"""
import os,sys

import rules as primitive_rules
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell

from utils import extract_phrase_from_parse,flatten,transform,write_f
from nltk.tree import Tree
from collections import Counter
import cPickle as pkl


class RuleNet(object):
  def __init__(self, config, batch_size=1, is_training=True):

    self.batch_size = batch_size
    self.seq_length = 50

    self.max_rule_size = 50000
    self.rule_embedding_size = 300
    self.init_scale = config.init_scale
    self.rules = primitive_rules.rules
    print ' [**] RuleNet loaded'

  def create_model(self,
      neural_model,
      vocab_size=40000,embedding_size=300,
      max_per_rule=3
      ):

    self.vocab_size = vocab_size #50000
    self.embedding_size = embedding_size #300


    rule_types_filename = 'rule_type.pkl'
    rule_types_out_filename = 'rule_type.txt'

    if not os.path.exists(rule_types_filename):
      train_data, valid_data, test_data = neural_model.load_data_to_text()
      data = train_data + valid_data + test_data
      print 'Nuhmber of data',len(data),len(train_data), len(valid_data), len(test_data)

      print 'Generating rule types...'
      sent_all = [p for p,h,l in data]
      sent_all += [h for p,h,l in data]
      print 'Sentences %d'%(len(sent_all)) #,len(premise_all),len(hypothesis_all))

      rule_all = []
      for sidx,sent_parse in enumerate(sent_all):
        sent = ' '.join(Tree.fromstring(str(sent_parse)).leaves()).lower()
        phrases = extract_phrase_from_parse(str(sent_parse))


        for rule_type,rule in self.rules.items():
          if rule_type == 'ppdb':
            for pos, phrase in phrases:
              if phrase in rule:
                for action_type, rhs_dict in rule[phrase].items():
                  if len(rhs_dict) == 0: continue
                  for rhs, pos_list in rhs_dict.items()[:max_per_rule]:
                    #print rule_type,action_type, phrase,pos,rhs
                    rule_all.append('|'.join([rule_type,action_type, phrase,pos,rhs]))
          elif rule_type == 'sick':
            for pos, phrase in phrases:
              if phrase in rule:
                for action_type, rhs_list in rule[phrase].items():
                  if len(rhs_list) == 0: continue
                  for rhs in list(set(rhs_list))[:max_per_rule]:
                    #print rule_type,action_type, phrase,pos,rhs
                    rule_all.append('|'.join([rule_type,action_type, phrase,pos,rhs]))
          elif rule_type == 'wordnet':
            for pos, phrase in phrases:
              for action_type,action_rule in rule.items():
                wn_result = action_rule(phrase,pos=pos,max_entities=1, max_lemmas=1)
                if len(wn_result) ==0: continue
                for rhs in list(set(wn_result)):
                  #print rule_type,action_type, phrase, pos, rhs
                  rule_all.append('|'.join([rule_type,action_type, phrase,pos,rhs]))
          #TODO for hand rules

      rule_types = Counter(rule_all)
      print '\tTotal rule: %d tokens: '%(len(rule_all))
      pkl.dump(rule_types, open(rule_types_filename,'wb'))
    else:
      print '\tLoading rule types...',rule_types_filename
      rule_types = pkl.load(open(rule_types_filename,'rb'))

    with open(rule_types_out_filename,'w') as fout:
      fout.write('\n'.join(['%s\t%d'%(str(r), c) for r,c in rule_types.most_common()]))
    print '\tTotal rule: %d types: '%(len(rule_types))
    print '\t',rule_types.most_common(3)


    # creating model graph here
    self.rule_size = max(len(rule_types), self.max_rule_size)


    return

    with tf.variable_scope('RuleNet'):
      self.input_context = tf.placeholder(
          shape=(self.batch_size, self.seq_length), dtype=tf.int32, name='input_context')
      self.target_rule = tf.placeholder(
          shape=(self.batch_size, self.rule_size), dtype=tf.int32, name='input_rule')
#       self.rule_embedding = np.random.normal(
 #          0.0, self.init_scale, [self.rule_size, self.rule_embedding_size])
      self.word_embedding = tf.get_variable('embedding',
          [self.vocab_size, self.embedding_size],
          dtype=tf.float32,trainable=False,
          initializer=None)
      self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_size])
      self.embedding_init = self.word_embedding.assign(self.embedding_placeholder)

      self.emb_word = tf.nn.embedding_lookup(self.word_embedding, self.input_context)
      #self.emb_rule = tf.nn.embedding_lookup(self.rule_embedding, self.input_rule)

      w = tf.get_variable('w', [self.embedding_size, self.rule_size], dtype=tf.float32)
      b = tf.get_variable('b', [self.rule_size], dtype=tf.float32)
      self.logits = tf.nn.xw_plus_b(self.emb_word, w, b)
      self.probs  = tf.nn.softmax(self.logits)
      # p(cnt | rule, context)

      loss = seq2seq.sequence_loss_by_example(
        [self.logits],
        [tf.reshape(self.target_rule, [-1])],
        [tf.ones([self.batch_size * self.seq_length])],
        self.rule_size)
      self.cost = tf.reduce_sum(loss) / self.batch_size #/ self.seq_length
      #self.final_state = last_state
      #self.lr     = tf.Variable(0.0, trainable = False)
      tvars     = tf.trainable_variables()
      grads, _    = tf.clip_by_global_norm(tf.gradients(self.cost, tvars, aggregation_method=2), 5.0)
      optimizer   = tf.train.AdamOptimizer(1e-3)
      self.train_op = optimizer.apply_gradients(zip(grads, tvars))


  def generate_sentences(self,
      data, rule_cond, order_cond, rule_apply,
      write, limit=False, verbose=False,
      max_per_rule = 5):

    #rule_prior = self.rule_model.get_rule_prior()
    #write.write('#RULE\tTRAN/SFORM\tGENERATED\tHYPOTHESIS\tPREMISE\tAPPLY\tR1\tR2\tR3\n')

    generated_sents = []
    for sidx,(prem_parse, hypo_parse, label) in enumerate(data):

      prem = ' '.join(prem_parse.leaves()).lower()
      hypo = ' '.join(hypo_parse.leaves()).lower()
      if rule_apply =='premise':
        sent_parse = prem_parse
        sent = prem
      elif rule_apply == 'hypothesis':
        sent_parse = hypo_parse
        sent = hypo
      phrases = extract_phrase_from_parse(sent_parse)
      if verbose: print 'Label %d, SENT(%s): %s'%(label, rule_apply, sent, sent_parse)


      for rule_type,rule in self.rules.items():
        if rule_type == 'ppdb' and rule_cond in ['ppdb', 'all']:
          for pos, phrase in phrases:
            if phrase in rule:

              for action_type, rhs_dict in rule[phrase].items():
                if len(rhs_dict) == 0: continue
                rhs_set = []
                for rhs, pos_list in rhs_dict.items()[:max_per_rule]:

                  r1,r2,r3 = self.order_logic(label, 'entail', rule_apply)
                  if r3 is None: continue
                  if rhs in rhs_set: continue
                  else: rhs_set.append(rhs)

                  output = transform(sent,phrase,rhs)
                  if output == sent: continue
                  generated_sents.append(
                      (prem,hypo,rule_apply,output,r1,r2,r3,rule_type,action_type,phrase,rhs))

        elif rule_type == 'sick' and rule_cond in ['sick', 'all']:
          for pos, phrase in phrases:
            if phrase in rule:
              for action_type, rhs_list in rule[phrase].items():
                if len(rhs_list) == 0: continue
                rhs_set = []
                for rhs in list(set(rhs_list))[:max_per_rule]:

                  r1,r2,r3 = self.order_logic(label, action_type, rule_apply)
                  if r3 is None: continue
                  if rhs in rhs_set: continue
                  else: rhs_set.append(rhs)

                  output = transform(sent,phrase,rhs)
                  if output == sent: continue
                  generated_sents.append(
                      (prem,hypo,rule_apply,output,r1,r2,r3,rule_type,action_type,phrase,rhs))

        elif rule_type == 'wordnet' and rule_cond in ['wordnet', 'all']:
          for pos, phrase in phrases:
            for action_type,action_rule in rule.items():
              wn_result = action_rule(phrase,pos=pos,max_entities=1, max_lemmas=1)
              if len(wn_result) ==0: continue
              r3 = None
              if action_type == 'anto': r1,r2,r3 = self.order_logic(label, 'contradict', rule_apply)
              else: r1,r2,r3 = self.order_logic(label, 'entail', rule_apply)
              if r3 is None: continue

              for rhs in list(set(wn_result)):

                output = transform(sent,phrase,rhs)
                if output == sent: continue
                generated_sents.append(
                    (prem,hypo,rule_apply,output,r1,r2,r3,rule_type,action_type,phrase,rhs))

        elif rule_type == 'hand' and rule_cond in ['hand', 'all']:
          for action_type,action_rule in rule.items():
            wn_result = action_rule(sent)
            if len(wn_result) ==0: continue
            r3 = None
            r1,r2,r3 = self.order_logic(label, 'contradict', rule_apply)
            if r3 is None: continue
            output = wn_result
            if output == sent: continue
            generated_sents.append(
                (prem,hypo,rule_apply,output,r1,r2,r3,rule_type,action_type,'',''))
            #print sent, output,r1,r2,r3




    if verbose:
      print '\t[%s/%s/%s] %d generated from rules'%(rule_apply,rule_cond,order_cond,len(generated_sents))

    generated_sents_ordered = []
    for generated in generated_sents:
      prem,hypo,rule_apply,output,l1,l2,l3,rule_type,action_type,phrase,rhs = generated
      if rule_apply =='premise':
        if order_cond in ['first','all']:
          label = int(l2) #[0,0,0]
          generated_sents_ordered.append((prem,output,label))
        if order_cond in ['second','all']:
          label = int(l3) #[0,0,0]
          generated_sents_ordered.append((output,hypo,label))
      elif rule_apply == 'hypothesis':
        if order_cond in ['first','all']:
          label = int(l2) #[0,0,0]
          generated_sents_ordered.append((hypo,output,label))
        if order_cond in ['second','all']:
          label = int(l3)# [0,0,0]
          generated_sents_ordered.append((prem,output,label))

    if verbose:
      print '\t[%s/%s/%s] %d generated from orders'%(rule_apply,rule_cond,order_cond,len(generated_sents_ordered))

    if limit:
      generated_sents_ordered = generated_sents_ordered[:limit]
    if write:
      for gen in generated_sents:
       write_f(gen, verbose=verbose, fwrite=write)
      #print 'Saved %d examples: %s'%(len(generated_sents),write.name)
    #import pdb; pdb.set_trace()

    return generated_sents_ordered






  def run_epoch(self, session, data, is_training=True):
    epoch_size, id_to_data = bucket_shuffle(data)
    for step in enumerate(epoch_size):
      (id,(x,y)) = next(id_to_data)
      m = models[id]
      print x['premise'], x['hypothesis']
      #session.run([m.final_representation, m.logits] )


  def order_logic(self, label, action_type, rule_apply):
    """
     # P-R1-H & H-R2-H' => P-R3-H'
     # P-R1-H & P-R2-P' => P'-R3-H
    # R1: 0 (neutral) 1(entailment) 2(contradiction)
    # R2: (entail, contradict, neutral)
    """
    r3 = None
    r2 = 1 if action_type=='entail' else 2 if action_type=='contradict' else 0
    #import pdb; pdb.set_trace()
    if rule_apply == 'hypothesis':
      if label==1:
        if action_type=='entail': r3 = 1
        elif action_type=='contradict': r3 = 2
        elif action_type=='neutral': r3 = 0
        else: print 'ERROR',action_type,rule_apply,label,r3; sys.exit(1)
      elif label==2:
        if action_type in ['entail','contradict']: r3 = None
        elif action_type in ['neutral']: r3=0
        else: print 'ERROR',action_type,rule_apply,label,r3; sys.exit(1)
      elif label==0: r3 = 0
      else: print 'ERROR',action_type,rule_apply,label,r3; sys.exit(1)
    elif rule_apply == 'premise':
      if label==1 or label==2:
        if action_type in ['entail','contradict']: r3=None
        elif action_type in ['neutral']: r3=0
        else: print 'ERROR',action_type,rule_apply,label,r3; sys.exit(1)
      elif label==0: r3 = 0
      else: print 'ERROR',action_type,rule_apply,label,r3; sys.exit(1)
    else: print 'ERROR',action_type,rule_apply,label,r3; sys.exit(1)

    return label,r2, r3



  def train(self):
    self.create_model()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  def str2bool(v):
    if v.lower() in ('yes','true','t', 'l'): return True
    elif v.lower() in ('no','false','f'): return False
    else: return v

  parser.register('type','bool',str2bool) # add type keyword to registries
  parser.add_argument("--dataset")
  parser.add_argument("--project_path")
  parser.add_argument("--model_path")
  parser.add_argument("--data_path")
  parser.add_argument("--model",default="adversarial")

  parser.add_argument("--subsample",type='bool',default=False)     #False/True
  parser.add_argument("--is_test",action='store_true',default=False)
  parser.add_argument("--reload",action='store_true',default=False)
  args = parser.parse_args()

  model = AdversarialLearning(args)
  model.load_model()
  pprint.pprint(vars(args),width=1)

  if not args.is_test:
    model.train(verbose=True, write=True)



def bucket_shuffle(dict_data):
  # zip each data tuple with it's bucket id.
  # return as a randomly shuffled iterator.
  id_to_data =[]
  for x, data in dict_data.items():
    id_to_data += list(zip([x]*len(data), data))

  shuffle(id_to_data)

  return len(id_to_data), iter(id_to_data)




