import argparse, sys, os, time, logging, threading, traceback
import numpy as np
import tensorflow as tf
import _pickle as pkl
import sys
from multiprocessing import Queue, Process

from Config import Config
from model import model
from data_iterator import TextIterator, preparedata
from dataprocess.vocab import Vocab
import utils

_REVISION = 'flatten'

parser = argparse.ArgumentParser(description="training options")

parser.add_argument('--load-config', action='store_true', dest='load_config', default=False)
parser.add_argument('--gpu-num', action='store', dest='gpu_num', default=0, type=int)
parser.add_argument('--train-test', action='store', dest='train_test', default='train', choices=['train', 'test'])
parser.add_argument('--weight-path', action='store', dest='weight_path', required=True)
parser.add_argument('--restore-ckpt', action='store_true', dest='restore_ckpt', default=False)
parser.add_argument('--retain-gpu', action='store_true', dest='retain_gpu', default=False)

parser.add_argument('--debug-enable', action='store_true', dest='debug_enable', default=False)

args = parser.parse_args()

DEBUG = args.debug_enable
if not DEBUG:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def debug(s):
    if DEBUG:
        print(s)
    pass

class Train:

    def __init__(self, args):
        if utils.valid_entry(args.weight_path) and not args.restore_ckpt\
                and args.train_test != 'test':
            raise ValueError('process running or finished')

        gpu_lock = threading.Lock()
        gpu_lock.acquire()
        def retain_gpu():
            if args.retain_gpu:
                with tf.Session():
                    gpu_lock.acquire()
            else:
                pass

        lockThread = threading.Thread(target=retain_gpu)
        lockThread.start()
        try:
            self.args = args
            config = Config()

            self.args = args
            self.weight_path = args.weight_path

            if args.load_config == False:
                config.saveConfig(self.weight_path + '/config')
                print('default configuration generated, please specify --load-config and run again.')
                gpu_lock.release()
                lockThread.join()
                sys.exit()
            else:
                if os.path.exists(self.weight_path + '/config'):
                    config.loadConfig(self.weight_path + '/config')
                else:
                    raise ValueError('No config file in %s' % self.weight_path)

            if config.revision != _REVISION:
                raise ValueError('revision dont match: %s over %s' % (config.revision, _REVISION))

            vocab = Vocab()
            vocab.load_vocab_from_file(os.path.join(config.datapath, 'vocab.pkl'))
            config.vocab_dict = vocab.word_to_index
            with open(os.path.join(config.datapath, 'label2id.pkl'), 'rb') as fd:
                _ = pkl.load(fd)
                config.id2label = pkl.load(fd)
                _ = pkl.load(fd)
                config.id2weight = pkl.load(fd)

            config.class_num = len(config.id2label)
            self.config = config

            self.train_data = TextIterator(os.path.join(config.datapath, 'trainset.pkl'), self.config.batch_sz,
                                           bucket_sz=self.config.bucket_sz, shuffle=True)
            config.n_samples = self.train_data.num_example
            self.dev_data = TextIterator(os.path.join(config.datapath, 'devset.pkl'), self.config.batch_sz,
                                         bucket_sz=self.config.bucket_sz, shuffle=False)

            self.test_data = TextIterator(os.path.join(config.datapath, 'testset.pkl'), self.config.batch_sz,
                                         bucket_sz=self.config.bucket_sz, shuffle=False)

            self.data_q = Queue(10)

            self.model = model(config)

        except Exception as e:
            traceback.print_exc()
            gpu_lock.release()
            lockThread.join()
            exit()

        gpu_lock.release()
        lockThread.join()
        if utils.valid_entry(args.weight_path) and not args.restore_ckpt\
                and args.train_test != 'test':
            raise ValueError('process running or finished')

    def get_epoch(self, sess):
        epoch = sess.run(self.model.on_epoch)
        return epoch

    def run_epoch(self, sess, input_data: TextIterator, verbose=10):
        """Runs an epoch of training.

        Trains the model for one-epoch.

        Args:
            sess: tf.Session() object
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        total_steps = input_data.num_example // input_data.batch_sz
        total_loss = []
        total_w_loss = []
        total_ce_loss = []
        collect_time = []
        collect_data_time = []
        accuracy_collect = []
        step = -1
        dataset = [o for o in input_data]
        producer = Process(target=preparedata,
                           args=(dataset, self.data_q, self.config.max_wd_num, self.config.id2weight))
        producer.start()
        try:
            while True:
                step += 1
                start_stamp = time.time()
                data_batch = self.data_q.get()
                if data_batch is None:
                    break
                feed_dict = self.model.create_feed_dict(data_batch=data_batch, train=True)

                data_stamp = time.time()
                (accuracy, global_step, summary, opt_loss, w_loss, ce_loss, lr, _
                 ) = sess.run([self.model.accuracy, self.model.global_step, self.merged,
                               self.model.opt_loss, self.model.w_loss, self.model.ce_loss,
                               self.model.learning_rate, self.model.train_op],
                              feed_dict=feed_dict)
                self.train_writer.add_summary(summary, global_step)
                self.train_writer.flush()
                end_stamp = time.time()

                collect_time.append(end_stamp-start_stamp)
                collect_data_time.append(data_stamp-start_stamp)
                accuracy_collect.append(accuracy)
                total_loss.append(opt_loss)
                total_w_loss.append(w_loss)
                total_ce_loss.append(ce_loss)

                if verbose and step % verbose == 0:
                    sys.stdout.write('\r%d / %d : opt_loss = %.4f, w_loss = %.4f, ce_loss = %.4f, %.3fs/iter, %.3fs/batch'
                                     'lr = %f, accu = %.4f, b_sz = %d' % (
                        step, total_steps, np.mean(total_loss[-verbose:]),np.mean(total_w_loss[-verbose:]),
                        np.mean(total_ce_loss[-verbose:]), np.mean(collect_time), np.mean(collect_data_time), lr,
                        np.mean(accuracy_collect[-verbose:]), input_data.batch_sz))
                    collect_time = []
                    sys.stdout.flush()
                    utils.write_status(self.weight_path)
        except:
            traceback.print_exc()
            producer.terminate()
            exit()

        producer.join()

        sess.run(self.model.on_epoch_accu)

        return np.mean(total_ce_loss), np.mean(total_loss), np.mean(accuracy_collect)

    def fit(self, sess, input_data :TextIterator, verbose=10):
        """
        Fit the model.

        Args:
            sess: tf.Session() object
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """

        total_steps = input_data.num_example // input_data.batch_sz
        total_loss = []
        total_ce_loss = []
        collect_time = []
        step = -1
        dataset = [o for o in input_data]
        producer = Process(target=preparedata,
                           args=(dataset, self.data_q, self.config.max_wd_num, self.config.id2weight))
        producer.start()
        try:
            while True:
                step += 1
                data_batch = self.data_q.get()
                if data_batch is None:
                    break
                feed_dict = self.model.create_feed_dict(data_batch=data_batch, train=False)

                start_stamp = time.time()
                (global_step, summary, ce_loss, opt_loss,
                 ) = sess.run([self.model.global_step, self.merged, self.model.ce_loss,
                               self.model.opt_loss], feed_dict=feed_dict)

                self.test_writer.add_summary(summary, step+global_step)
                self.test_writer.flush()

                end_stamp = time.time()
                collect_time.append(end_stamp - start_stamp)
                total_ce_loss.append(ce_loss)
                total_loss.append(opt_loss)

                if verbose and step % verbose == 0:
                    sys.stdout.write('\r%d / %d: ce_loss = %f, opt_loss = %f,  %.3fs/iter' % (
                        step, total_steps, np.mean(total_ce_loss[-verbose:]),
                        np.mean(total_loss[-verbose:]), np.mean(collect_time)))
                    collect_time = []
                    sys.stdout.flush()
            print('\n')
        except:
            traceback.print_exc()
            producer.terminate()
            exit()
        producer.join()
        return np.mean(total_ce_loss), np.mean(total_loss)

    def predict(self, sess, input_data: TextIterator, verbose=10):
        """
        Args:
            sess: tf.Session() object
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        total_steps = input_data.num_example // input_data.batch_sz
        collect_time = []
        collect_pred = []
        label_id = []
        step = -1
        dataset = [o for o in input_data]
        producer = Process(target=preparedata,
                           args=(dataset, self.data_q, self.config.max_wd_num, self.config.id2weight))
        producer.start()
        try:
            while True:
                step += 1
                data_batch = self.data_q.get()
                if data_batch is None:
                    break
                feed_dict = self.model.create_feed_dict(data_batch=data_batch, train=False)

                start_stamp = time.time()
                pred = sess.run(self.model.prediction, feed_dict=feed_dict)
                end_stamp = time.time()
                collect_time.append(end_stamp - start_stamp)

                collect_pred.append(pred)
                label_id += data_batch[1].tolist()
                if verbose and step % verbose == 0:
                    sys.stdout.write('\r%d / %d: , %.3fs/iter' % (
                        step, total_steps, np.mean(collect_time)))
                    collect_time = []
                    sys.stdout.flush()
            print('\n')
        except:
            traceback.print_exc()
            producer.terminate()
            exit()
        producer.join()
        res_pred = np.concatenate(collect_pred, axis=0)
        return res_pred, label_id

    def test_case(self, sess, data, onset='VALIDATION'):
        print('#' * 20, 'ON ' + onset + ' SET START ', '#' * 20)
        print("=" * 10 + ' '.join(sys.argv) + "=" * 10)
        epoch = self.get_epoch(sess)
        ce_loss, opt_loss = self.fit(sess, data)
        pred, label = self.predict(sess, data)

        (prec, recall, overall_prec, overall_recall, _
         ) = utils.calculate_confusion_single(pred, label, len(self.config.id2label))

        utils.print_confusion_single(prec, recall, overall_prec, overall_recall, self.config.id2label)
        accuracy = utils.calculate_accuracy_single(pred, label)

        print('%d th Epoch -- Overall %s accuracy is: %f' % (epoch, onset, accuracy))
        logging.info('%d th Epoch -- Overall %s accuracy is: %f' % (epoch, onset, accuracy))

        print('%d th Epoch -- Overall %s ce_loss is: %f, opt_loss is: %f' % (epoch, onset, ce_loss, opt_loss))
        logging.info('%d th Epoch -- Overall %s ce_loss is: %f, opt_loss is: %f' % (epoch, onset, ce_loss, opt_loss))
        print('#' * 20, 'ON ' + onset + ' SET END ', '#' * 20)
        return accuracy, ce_loss

    def train_run(self):
        logging.info('Training start')
        logging.info("Parameter count is: %d" % self.model.param_cnt)
        if not args.restore_ckpt:
            self.remove_file(self.args.weight_path + '/summary.log')
        saver = tf.train.Saver(max_to_keep=30)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:

            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(self.args.weight_path + '/summary_train',
                                                 sess.graph)
            self.test_writer = tf.summary.FileWriter(self.args.weight_path + '/summary_test')

            sess.run(tf.global_variables_initializer())
            if args.restore_ckpt:
                saver.restore(sess, self.args.weight_path + '/classifier.weights')
            best_loss = np.Inf
            best_accuracy = 0
            best_val_epoch = self.get_epoch(sess)

            for _ in range(self.config.max_epochs):

                epoch = self.get_epoch(sess)
                print("=" * 20 + "Epoch ", epoch, "=" * 20)
                ce_loss, opt_loss, accuracy = self.run_epoch(sess, self.train_data, verbose=10)
                print('')
                print("Mean ce_loss in %dth epoch is: %f, Mean ce_loss is: %f,"%(epoch, ce_loss, opt_loss))
                print('Mean training accuracy is : %.4f' % accuracy)
                logging.info('Mean training accuracy is : %.4f' % accuracy)
                logging.info("Mean ce_loss in %dth epoch is: %f, Mean ce_loss is: %f,"%(epoch, ce_loss, opt_loss))
                print('=' * 50)
                val_accuracy, val_loss = self.test_case(sess, self.dev_data, onset='VALIDATION')
                test_accuracy, test_loss = self.test_case(sess, self.test_data, onset='TEST')
                self.save_loss_accu(self.args.weight_path + '/summary.log', train_loss=ce_loss,
                                    valid_loss=val_loss, test_loss=test_loss,
                                    valid_accu=val_accuracy, test_accu=test_accuracy, epoch=epoch)
                if best_accuracy < val_accuracy:
                    best_accuracy = val_accuracy
                    best_val_epoch = epoch
                    if not os.path.exists(self.args.weight_path):
                        os.makedirs(self.args.weight_path)
                    logging.info('best epoch is %dth epoch' % best_val_epoch)
                    saver.save(sess, self.args.weight_path + '/classifier.weights')
                else:
                    b_sz = self.train_data.batch_sz//2
                    max_b_sz = max([b_sz, self.config.batch_sz_min])
                    buck_sz = self.train_data.bucket_sz * 2
                    buck_sz = min([self.train_data.num_example, buck_sz])
                    self.train_data.batch_sz = max_b_sz
                    self.train_data.bucket_sz = buck_sz

                if epoch - best_val_epoch > self.config.early_stopping:
                    logging.info("Normal Early stop")
                    break
        utils.write_status(self.weight_path, finished=True)
        logging.info("Training complete")

    def test_run(self):

        saver = tf.train.Saver(max_to_keep=30)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        with tf.Session(config=config) as sess:
            self.merged = tf.summary.merge_all()
            self.test_writer = tf.summary.FileWriter(self.args.weight_path + '/summary_test')

            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self.args.weight_path + '/classifier.weights')

            self.test_case(sess, self.test_data, onset='TEST')

    def main_run(self):

        if not os.path.exists(self.args.weight_path):
            os.makedirs(self.args.weight_path)
        logFile = self.args.weight_path + '/run.log'

        if self.args.train_test == "train":

            try:
                os.remove(logFile)
            except OSError:
                pass
            logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
            debug('_main_run_')
            self.train_run()
            self.test_run()
        else:
            logging.basicConfig(filename=logFile, format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
            self.test_run()

    @staticmethod
    def save_loss_accu(fileName, train_loss, valid_loss,
                       test_loss, valid_accu, test_accu, epoch):
        with open(fileName, 'a') as fd:
            fd.write('%3d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' %
                     (epoch, train_loss, valid_loss,
                      test_loss, valid_accu, test_accu))

    @staticmethod
    def remove_file(fileName):
        if os.path.exists(fileName):
            os.remove(fileName)

if __name__ == '__main__':
    trainer = Train(args)
    trainer.main_run()

