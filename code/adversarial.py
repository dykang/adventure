import os,sys,argparse,pprint,random,time
from datetime import datetime
from sys import argv
import numpy as np
from collections import defaultdict,OrderedDict,Counter
import cPickle as pkl
import tensorflow as tf
from nltk.tree import Tree
from nltk.tokenize import word_tokenize

from learning import Learning
from utils import write_f,transform, extract_phrase_from_parse,traverseTree,flatten
sys.path.insert(1,os.path.join(sys.path[0], '..'))
from neu.utils import progress

from neu.epoch import bucket_shuffle

class AdversarialLearning(Learning):
  def __init__(self,args):

    self.checkpoint_dir = '/data/tf/ckpts'
    self.log_dir = '/data/tf/logs'
    self.result_dir = 'results'

    self._attrs = ['generator','rule','rule_apply', 'order', 'subsample', 'ratio'] #'gan', ,'parse'
    Learning.__init__(self,
        model=args.model, dataset=args.dataset,
        model_path=args.model_path,data_path=args.data_path,
        generator=args.generator, rule=args.rule,
        gan=args.gan, parse=args.parse,
        rule_apply=args.rule_apply,order=args.order,
        subsample=args.subsample, ratio=args.ratio,
        reload=args.reload, beam=args.beam)



    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth=True
    tf_config.intra_op_parallelism_threads=15
    tf_config.inter_op_parallelism_threads=15
    self.sess = tf.InteractiveSession(config=tf_config)



  def create_model(self, is_training=False):

    self.sess = self.disc_model.sess

    print '[[%s]] Creating model...'%(self.name)
    self.rule_model.create_model(
        self.disc_model,
        max_per_rule = 3, vocab_size=40000 ,
        embedding_size = 300)

    self.initialize()
    print '[[%s]] Created...'%(self.name)


  def convert_batch_to_text(self,data):
    bid,x,y = data
    batch = []
    batch_parse = []
    for p,h,l in zip(x['premise'],x['hypothesis'],y):
      p = ' '.join([self.vocab.id_token[w] for w in p if w > 0])
      h = ' '.join([self.vocab.id_token[w] for w in h if w > 0])
      l  = np.argmax(l)
      batch.append((p,h,l))
    for p,h,l in zip(x['premise_parse'],x['hypothesis_parse'],y):
      l  = np.argmax(l)
      batch_parse.append((p,h,l))
    return bid,batch,batch_parse



  def generate_z(self,batch_x,write=False, verbose=True):
    bid,x,x_parse = self.convert_batch_to_text(batch_x)
    if verbose: print 'Number of x: %d'%(len(x))
    z, z_gen,z_rule = [],[], []
    if self.generator:
      z_gen = self.generate_with_seq2seq(bid, x, write=write, verbose=verbose)
      if verbose: print '\tNumber of z from seq2seq %d'%(len(z_gen))
    if self.rule:
      z_rule = self.generate_with_rulenet(bid, x_parse, write=write, verbose=verbose)
      if verbose: print '\tNumber of z from rulenet %d'%(len(z_rule))
    z = z_gen + z_rule
    if verbose: print 'Number of z: %d'%(len(z))

    # (1) label/data balancing
    zx = self.balancing(z,x, ratio=self.ratio, verbose=verbose)
    if verbose: print 'Number of zx: %d'%(len(zx))

    # (2) batching
    train_data, train_stats = self.load_batch(zx, bid)
    if verbose:
      print 'Buckets in train (zx): %s'%(
          ' '.join('%d:%d'%(bid,len(sents)*self.batch_size) for bid,sents in train_data.items()))
      print train_stats
    return train_data


  def categorical_label(self,label):
    try:
      onehot = [0,0,0]
      onehot[label] = 1
      return onehot #,category
    except Exception as e:
      print e
      import pdb; pdb.set_trace()

  def load_batch(self,zx, bid):
    bucket_id = bid
    max_len = self.buckets[-1] # last bucket size is the max
    output = defaultdict(list) #[[] for _ in range(len(self.buckets))]
    bucket_dict =  {i:defaultdict(list) for i in range(len(self.buckets))}
    stats = Counter()
    random.shuffle(zx)

    for line in zx:
      s1, s2, label = line
      s1 = word_tokenize(s1)
      s2 = word_tokenize(s2)
      label = self.categorical_label(label)
      if label is None:
        stats['bad_label'] += 1
        continue

      len_s1 = len(s1)
      len_s2 = len(s2)
      stats["max_len_premise"] = max(stats["max_len_premise"], len_s1)
      stats["max_len_hypothesis"] = max(stats["max_len_hypothesis"], len_s2)

      # drop item if either premise or hyp is too long
      if max_len and (len_s1 > max_len[0] or len_s2 > max_len[1]):
          stats['n_ignore_long'] += 1
          continue
      stats["num_examples"] += 1

      s1_ids = self.vocab.ids_for_tokens(s1, update=False)
      s2_ids = self.vocab.ids_for_tokens(s2, update=False)

      # pad using the bucket length tuples for premise and hypothesis
      s1_f = self.reader.pad_sentence(s1_ids, pad_length=self.buckets[bucket_id][0], pad_id=self.vocab.PAD_ID)
      s2_f = self.reader.pad_sentence(s2_ids, pad_length=self.buckets[bucket_id][1], pad_id=self.vocab.PAD_ID)

      bucket_dict[bucket_id]["s1_batch"].append(s1_f)
      bucket_dict[bucket_id]["s2_batch"].append(s2_f)
      bucket_dict[bucket_id]["labels"].append(label)

      # flush batch
      if len(bucket_dict[bucket_id]["s1_batch"]) == self.batch_size:
        sents = {
          "premise": np.asarray(bucket_dict[bucket_id]["s1_batch"]).reshape(self.batch_size, self.buckets[bucket_id][0]),
          "hypothesis": np.asarray(bucket_dict[bucket_id]["s2_batch"]).reshape(self.batch_size, self.buckets[bucket_id][1])}
        tar = np.asarray(bucket_dict[bucket_id]["labels"]).reshape(self.batch_size, 3)

        output[bucket_id].append((sents,tar))
        bucket_dict[bucket_id]["s1_batch"] = []
        bucket_dict[bucket_id]["s2_batch"] = []
        bucket_dict[bucket_id]["labels"] = []
    if len(output[bucket_id]) == 0:
      print 'Emtpy!'
      import pdb; pdb.set_trace()
    return  output, stats



  def balancing(self,z,x, ratio=1, verbose=False):
    # load balancing
    x_labels = [xone[2] for xone in x ]
    if verbose: print 'X:',Counter(x_labels).most_common()
    z_labels = [zone[2] for zone in z ]
    if verbose: print 'Z:',Counter(z_labels).most_common(),
    z_balanced = []
    z_dic = defaultdict(list)
    z_min = np.inf
    for l,c in Counter(z_labels).most_common(): z_min = min(z_min,c)
    for zone in z: z_dic[zone[2]].append(zone)
    for l,zones in z_dic.items():
      for zone in zones[:z_min]:
        z_balanced.append(zone)
    z_balanced_labels = [zone[2] for zone in z_balanced ]
    if verbose: print '->',Counter(z_balanced_labels).most_common()

    # |z| = |x| * ratio
    random.shuffle(z_balanced)
    num_z = int(len(x) * ratio)
    z_balanced = z_balanced[:num_z]
    zx = x + z_balanced
    if verbose: print 'ZX: ',Counter([zxone[2] for zxone in zx]).most_common()
    return zx


  def generate_with_rulenet(self,bucket_id,data,write,limit=False,verbose=False):
    generated = []
    batch_size = len(data)

    if self.rule_apply in ['hypothesis', 'all']:
      outputs =  self.rule_model.generate_sentences(
            data, self.rule, self.order, 'hypothesis',
            write, verbose=False, max_per_rule=3)
      generated += outputs
    if self.rule_apply in ['premise', 'all']:
      outputs = self.rule_model.generate_sentences(
            data, self.rule, self.order, 'premise',
            write, verbose=False, max_per_rule=3)
      generated += outputs
    if verbose:
      print 'Total generated from rule generators',len(generated)
    if limit: generated = generated[:limit]
    return generated


  def get_label(self,gen_type):
    if self.dataset == 'scitail_1.0':
      label_dict = {'positive':1, 'negative':0, 'neutral':-1}
    elif self.dataset == 'snli_1.0':
      label_dict = {'positive':1, 'negative':2, 'neutral':0}
    return label_dict[gen_type]

  def generate_with_seq2seq(self, bid,data, write=False, limit=False, verbose=False):
    batch_size = len(data)
    generated = []
    if self.generator in ['neutral', 'all'] and self.dataset == 'snli_1.0':
      outputs = self.gen_neu_model.test_multiple_s2s(
          bid,data, self.get_label('neutral'), verbose=verbose, batch_size = batch_size)
      generated += outputs
      if verbose: print 'Generated (neutral):',len(outputs),self.generator

    if self.generator in ['positive', 'all']:
      outputs = self.gen_pos_model.test_multiple_s2s(
          bid,data, self.get_label('positive'), verbose=verbose, batch_size = batch_size)
      generated += outputs
      if verbose: print 'Generated (positive):',len(outputs),self.generator

    if self.generator in ['negative', 'all']:
      outputs = self.gen_neg_model.test_multiple_s2s(
          bid,data, self.get_label('negative'), verbose=verbose, batch_size = batch_size)
      generated += outputs
      if verbose: print 'Generated (negative):',len(outputs),self.generator

    if limit: generated = generated[:limit]
    if verbose:
      print 'Total generated from s2s generators',len(generated)

    if write:
      for gen in generated:
        write_f(gen, verbose=False, fwrite=write)
    return generated



  def step_disc(self, data, model, is_training=True, verbose=False):
    iters = 0
    accs, costs = .0, .0
    if is_training:
      for bid,d in data.items():
        for zx,zy in d:
          m = model[bid]
          acc, cost, _ = self.sess.run(
              [m.accuracy, m.cost, m.train_op],
              feed_dict={m.premise: zx["premise"],
                        m.hypothesis: zx["hypothesis"],
                        m.targets: zy})
          if verbose: print bid, acc, cost
          iters += 1
          costs += cost
          accs += acc
      return costs / iters , acc / iters
    else:
      cost, acc = self.disc_model.run_epoch(self.sess, model, data, training=False, sample=3)
      return cost, acc

  def step_gen(self, data, cost_disc, self.disc_train, is_training=True, verbose=False):
    batch_size = len(data)
    generated = []
    if self.generator in ['neutral', 'all'] and self.dataset == 'snli_1.0':
      outputs = self.gen_neu_model.backward_from_disc(
          data, cost_disc, self.disc_train,  self.get_label('neutral'), verbose=verbose, batch_size = batch_size)

    if self.generator in ['positive', 'all']:
      outputs = self.gen_pos_model.backward_from_disc(
          data, cost_disc, self.disc_train,  self.get_label('positive'), verbose=verbose, batch_size = batch_size)

    if self.generator in ['negative', 'all']:
      outputs = self.gen_neg_model.backward_from_disc(
          data, cost_disc, self.disc_train,  self.get_label('negative'), verbose=verbose, batch_size = batch_size)

    return





  def test(self, data, verbose=False, write=True, output=False):
    print 'Testing...'
    # Load saved generated data and replace train_data
    self.batch_size = self.disc_model.config.batch_size
    self.vocab = self.disc_model.vocab
    self.buckets = self.disc_model.buckets

    self.create_model(is_training=False)
    self.disc_model.test(verbose=True, output=output) #, data=data)


  def train(self, infer=False, verbose=False, write=True, output=False):

    print 'Training...'
    # Load saved generated data and replace train_data
    self.batch_size = self.disc_model.config.batch_size
    self.vocab = self.disc_model.vocab
    self.buckets = self.disc_model.buckets

    self.create_model(is_training=True)
    self.disc_model.test(verbose=True, output=output)

    # read generated data and train GAN
    max_epoch = 70
    train_accs, valid_accs, test_accs = [], [], []

    train_buckets = self.disc_model.train_buckets
    val_buckets = self.disc_model.val_buckets
    test_buckets = self.disc_model.test_buckets
    print 'Buckets in train (%d) valid (%d) test (%d)'%(
        sum([len(train_buckets[bid])*self.batch_size for bid in range(len(self.disc_model.buckets))]),
        sum([len(val_buckets[bid])*self.batch_size for bid in range(len(self.disc_model.buckets))]),
        sum([len(test_buckets[bid])*self.batch_size for bid in range(len(self.disc_model.buckets))])
        )

    # While until L converges
    for epoch in range(max_epoch):
      if write:
        model_dir,attr_dir = self.get_model_dir()
        attr_dir = attr_dir.replace('/','_')
        adversarial_dir = os.path.join(self.model_path,model_dir,'generated',attr_dir)
        filename = os.path.join(adversarial_dir+'_train%d.txt'%(epoch))
        if os.path.exists(filename): os.remove(filename)
        write = open(filename, 'a')

      random.shuffle(train_buckets)
      step_size, id_to_data = bucket_shuffle(train_buckets)

      ##############################
      # start training
      ##############################
      print 'Epoch [%d/%d/%d]'%(epoch,step_size,max_epoch)
      costs, accuracies, gen_costs = .0, .0, .0
      iters = 0
      start_time = time.time()
      for step in range(step_size):
        bid,(x,y) = next(id_to_data)
        data = (bid,x,y)

        # generate z and load balacing with x
        zx_train = self.generate_z(data, write=write, verbose=False)
        # step for disc training
        cost_disc,acc_disc = self.step_disc(zx_train, self.disc_train, is_training=True, verbose=False)

        # step for gen training
        cost_gen = self.step_gen(zx_train, cost_disc, self.disc_train, is_training=True, verbose=False)

        costs += cost_disc
        gen_costs += cost_gen
        accuracies += acc_disc
        iters += 1

        progress(step * 1.0 / step_size,"acc: %.3f loss: (disc:%.3f, gen:%.3f) speed: %.0f ex/s"
          %(accuracies / iters, costs / iters, gen_costs / iters,  iters * self.batch_size / (time.time() - start_time)))
        break
      train_acc = accuracies / iters

      # check valid/test accuracy
      cost_valid, valid_acc = self.step_disc(val_buckets,self.disc_valid,is_training=False)
      cost_test, test_acc = self.step_disc(test_buckets,self.disc_test,is_training=False)
      print("> Epoch [%d]: Valid Acc: %.2f Test Acc: %.2f"%(epoch, valid_acc*100.0,test_acc*100.0))



      ##### Model Hooks #####
      if epoch > 0 and valid_acc >= max(valid_accs):
        date = "{:%m.%d.%H.%M}".format(datetime.now())
        with open("summary.txt", "a+") as sout:
          sout.write(" ".join([str(epoch+1),str(train_acc), str(valid_acc), str(test_acc), date, "\n"]))
        print ' [*] NEW BEST mode found %d'%(epoch)
        self.save(global_step = epoch)

      valid_accs.append(valid_acc)
      train_accs.append(train_acc)
      test_accs.append(test_acc)


      ##### Early Stopping #####
      if epoch > 5 and valid_acc < min(valid_accs[-1], valid_accs[-2], valid_accs[-3]):
        print '[*] EARLY STOPPING...'
        break


    # get final test output with the best dev model
    best_valid_epoch = np.argmax(valid_accs)
    print 'BEST EPOCH',best_valid_epoch
    print("Epoch: {} Train Acc: {}".format(epoch + 1, train_accs[best_valid_epoch]*100.0))
    print("Epoch: {} Valid Acc: {}".format(epoch + 1, valid_accs[best_valid_epoch]*100.0))
    print("Epoch: {} Test Acc: {}".format(epoch + 1, test_accs[best_valid_epoch]*100.0))

    with open('log_%s.txt'%(self.dataset),'a') as fout:
      _,attr_dir = self.get_model_dir()
      attr_dir = attr_dir.replace('/','_')
      fout.write('%s\t%d\t%.2f\t%.2f\t%.2f\n'%(
        attr_dir,best_valid_epoch,
        train_accs[best_valid_epoch]*100.0,
        valid_accs[best_valid_epoch]*100.0,
        test_accs[best_valid_epoch]*100.0))
      print 'saved to log.txt'




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

  parser.add_argument("--generator",type='bool',default=False)      #all/positive/negative/neutral/False
  parser.add_argument("--rule",type='bool',default=False)           #all/ppdb/sick/wordnet/False
  parser.add_argument("--gan",type='bool',default=False)     #False/True
  parser.add_argument("--parse",type='bool',default=False)     #False/True

  parser.add_argument("--rule_apply",default='hypothesis')
  parser.add_argument("--order",default='second')

  parser.add_argument("--subsample",type='bool',default=False)     #False/True
  parser.add_argument("--ratio",type=float,default=1.0)     #False/True

  parser.add_argument("--is_test",action='store_true',default=False)
  parser.add_argument("--reload",action='store_true',default=False)
  parser.add_argument("--beam",action='store_true',default=False)
  args = parser.parse_args()

  model = AdversarialLearning(args)
  model.load_model()
  pprint.pprint(vars(args),width=1)

  if not args.is_test:
    model.train(infer='test', verbose=True, write=False, output='output.txt')
  else:
    data = model.load_data_to_text(
        dataset='negation.SNLI.one', data_path='./')
        #dataset=args.is_test, data_path='%s/%s'%(args.data_path,args.is_test))
    model.test(
        data, verbose=True, write=True, output='output.txt')

