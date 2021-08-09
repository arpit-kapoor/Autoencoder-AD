# ========================================================= #
# Anomaly detection using Autoencoder (Tensorflow)
# ---------------------------------------------------------
# Author: Arpit Kapoor (kapoor.arpit97@gmail.com)
# ========================================================= #

import pickle 
import tensorflow as tf
import datetime
import numpy as np
import pandas as pd
import os
import datetime
import time
import argparse
from copy import deepcopy

from model import Autoencoder

"""
******************************* Trainer Class **********************************
"""
class AutoencoderTrainer(object):
    def __init__(self, data_path, cols, word2int, int2word, run_id, log_file ,run_dir, batch_size=100000, intermediate_dim=10, learning_rate=0.001):
        self.data_path = data_path
        self.run_id = run_id
        self.log_file = log_file
        self.run_dir = run_dir
        self.cols = cols
        self.batch_size = batch_size
        self.word2int = word2int
        self.int2word = int2word
        self.vocab_size = {col:len(words) for col, words in self.word2int.items()}

        self.input_size = sum([self.vocab_size[col] for col in self.cols])
        self.model = Autoencoder(intermediate_dim=intermediate_dim, original_dim=self.input_size)
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='test_loss')
        tf_log_dir = f'../tf_logs/train_{self.run_id}/'
        self.train_summary_writer = tf.summary.create_file_writer(tf_log_dir)

    
    @staticmethod
    def to_one_hot(data_point_index, vocab_size):
        """function to convert numbers to one hot vectors"""
        temp = np.zeros(vocab_size)
        temp[data_point_index] = 1
        return temp
    
    def loss(self, model, features):
        reconstruction_error = tf.square(tf.subtract(model(features), features))
        return reconstruction_error
    
    def preprocess_data(self, data, cols):
        features = []
        for col in cols:
            data[col] = data[col].str.lower()
        data = data.fillna('null')
        # Generate Training Data       
        data = data[cols].values.tolist()
        for data_word in data:
            features.append(np.concatenate([self.to_one_hot(self.word2int[col][data_word[idx]], self.vocab_size[col]) for idx, col in enumerate(cols)]))        
        processed_features = np.asarray(features).astype('float32')
        return processed_features    
    
    @tf.function
    def train_step(self, features):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(self.loss(self.model, features))
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
    
    @tf.function
    def val_step(self, features):
        loss = tf.reduce_mean(self.loss(self.model, features))
        self.val_loss(loss)
    
    def train(self, num_epochs=100, checkpoint_interval=10):
        
        model_path = f'{self.run_dir}/models/'
        best_model_path = os.path.join(model_path, 'best_model')
        best_loss = np.finfo('f').max
        
        descriptor = range(num_epochs)
        with self.train_summary_writer.as_default():
            with tf.summary.record_if(True):
                print('>>>>>>>>>>>>>> start training <<<<<<<<<<<<<<')
                for epoch in descriptor:
                    start_time = time.time()
                    write_log(f'>>>>>>>>>>>>>> EPOCH {epoch+1} <<<<<<<<<<<<<<', self.log_file)
                    print(f'>>>>>>>>>>>>>> EPOCH {epoch+1} <<<<<<<<<<<<<<')
                    print(f'>>>>>>> start time: {str(datetime.datetime.now())}')
                    write_log('>>>>>>> running training steps ', self.log_file)
                    print('>>>>>>> running training steps ')
                    data_gen = pd.read_csv(self.data_path['train_path'], dtype=str, chunksize=self.batch_size, iterator=True)
                    for idx, data in enumerate(data_gen):
                        # print(f'>>>>>>> epoch {epoch+1} train step {idx+1} started')
                        features = self.preprocess_data(data, self.cols)
                        self.train_step(features)
                        # print(f'>>>>>>> epoch {epoch+1} step {idx+1} finished')
                    
                    train_time = str(datetime.timedelta(seconds=int(time.time()-start_time)))
                    val_time = time.time()
                    write_log('>>>>>>> running validation steps ', self.log_file)
                    print('>>>>>>> running validation steps ')
                    data_gen = pd.read_csv(self.data_path['test_path'], dtype=str, chunksize=self.batch_size, iterator=True)
                    for idx, data in enumerate(data_gen):
                        # print(f'>>>>>>> epoch {epoch+1} validation step {idx+1} started')
                        features = self.preprocess_data(data, self.cols)
                        self.val_step(features)
                    
                    train_loss = self.train_loss.result()
                    val_loss = self.val_loss.result()
                    
                    tf.summary.scalar('train_loss', train_loss, step=epoch)
                    tf.summary.scalar('val_loss', val_loss, step=epoch)
                    
                    # Save if checkpoint reached
                    if (epoch+1) % checkpoint_interval == 0:
                        save_path = os.path.join(model_path, f'model_checkpoint_{epoch+1}_{val_loss:.4f}')
                        self.model.save(save_path)
                        write_log(f'>>>>>>> checkpoint saved to: {save_path}', self.log_file)
                        print(f'>>>>>>> checkpoint saved to: {save_path}')
                    
                    # Save Best Model
                    if val_loss < best_loss:
                        best_loss = val_loss
                        self.model.save(best_model_path)
                        write_log(f'>>>>>>> Best checkpoint saved to: {best_model_path}', self.log_file)
                        print(f'>>>>>>> Best checkpoint saved to: {best_model_path}')
                    
                    total_time = str(datetime.timedelta(seconds=int(time.time()-start_time)))
                    val_time = str(datetime.timedelta(seconds=int(time.time()-val_time)))
                    template = "*"*106 +'\n* Epoch {}/{}, Train loss: {:.5f}  Val loss: {:.5f} Time:- train: {}s val: {}s total: {}s *\n'+ "*"*106
                    
                    write_log("\n"+template.format(epoch+1, num_epochs, self.train_loss.result(), self.val_loss.result(), train_time, val_time, total_time), self.log_file)
                    print(template.format(epoch+1, num_epochs, self.train_loss.result(), self.val_loss.result(), train_time, val_time, total_time))
                    
                    # Reset the metrics for the next epoch
                    self.train_loss.reset_states()
                    self.val_loss.reset_states()

        return self.model

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def predict(self, data_path, out_path, additional_cols=[], batch_size=100000):
        data_gen = pd.read_csv(data_path, dtype=str, chunksize=batch_size, iterator=True)
        columns = additional_cols + self.cols + ["feat_embd", "loss"]
        data_out = pd.DataFrame([], columns=columns)

        print('>>>>>>> Prediction <<<<<<')
        write_log('>>>>>>> Prediction <<<<<<', self.log_file)
        for idx, data in enumerate(data_gen):
            features_in = self.preprocess_data(deepcopy(data), trainer.cols)
            loss = self.loss(self.model, features_in).numpy().mean(axis=1)
            # print(loss)
            feat_embd = self.model.encoder(features_in).numpy()
            data["loss"] = loss
            data["feat_embd"] = feat_embd.tolist()
            if idx == 0:
                data[columns].to_csv(out_path, header=True, index=False)
            else:
                data[columns].to_csv(out_path, mode='a', header=False, index=False)
            if (idx+1) % 10 == 0:
                print(f">>>>>> Prediction done for : {idx+1} batches")
                write_log(f">>>>>> Prediction done for : {idx+1} batches", self.log_file)

        print(">>>>>>> Prediction Done <<<<<<<")
        write_log(">>>>>>> Prediction Done <<<<<<<", self.log_file)

        return data_out

"""
****************************** Logging Function ********************************
"""
def write_log(msg, log_file):
    with open(log_file, 'a') as f:
        log_time = datetime.datetime.now()
        print(f'INFO: {log_time} -- {msg}', file=f) 

"""
************************************ Main **************************************
"""
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Train Autoencoder in Tensorflow')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--run-id', type=str)
    parser.add_argument('--vec-size', type=int, default=20)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    
    parser.add_argument('--train-data', type=str, default='../data/spm_data/csv_file/spm_train.csv')
    parser.add_argument('--val-data', type=str, default='../data/spm_data/csv_file/spm_test.csv')
    parser.add_argument('--test-data', type=str)

    parser.add_argument('--out-path', type=str)
    parser.add_argument('--load-model', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=100000)

    #---------------------------------------------------------------------------

    args = parser.parse_args()

    int2word = pickle.load(open('../data/spm_data/pickle_files/int2word.pickle', 'rb'))
    word2int = pickle.load(open('../data/spm_data/pickle_files/word2int.pickle', 'rb'))

    # Takes care Run Id and log_file
    if args.run_id is None:
        args.run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        
    run_dir = f'../runs/run_{args.run_id}'
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    log_file = os.path.join(run_dir, 'logs.txt')
    write_log(str(args), log_file)

    # Data Path
    data_path = {
        'train_path': args.train_data,
        'test_path': args.val_data
    }

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # >>>>>>>>>>>>>>>>>>>>>>> Make Changes to Columns here  <<<<<<<<<<<<<<<<<<<<
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    identifier = []
    columns = []

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    # Trainer Object
    trainer = AutoencoderTrainer(data_path, columns, word2int, int2word, run_id=args.run_id, intermediate_dim=args.vec_size, learning_rate=args.lr, batch_size=args.batch_size, log_file=log_file, run_dir=run_dir)
    
    # Load trained model
    if args.load_model:
        trainer.load_model(args.load_model)
    
    # Train model
    if args.train:
        autoencoder = trainer.train(num_epochs=args.num_epochs, checkpoint_interval=args.save_interval)
    
    # Run inference
    if args.test:
        if args.test_data is None:
            raise(Exception('Test data path missing!'))
        trainer.predict(args.test_data, out_path=args.out_path,  additional_cols=identifier, batch_size=args.batch_size)
