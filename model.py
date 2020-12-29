""" Sketch-RNN Implementation in Keras - Model"""
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers.core import RepeatVector
from keras.layers import Dense, LSTM, CuDNNLSTM, Bidirectional, Lambda
from keras.activations import softmax, exponential, tanh
from keras import backend as K
from keras.initializers import RandomNormal
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD
import numpy as np
import random

#defining hyperparameters
def get_default_hparams():
    """ Return default hyper-parameters """
    params_dict = {
        # Experiment Params:
        'is_training': True,  # train mode (relevant only for accelerated LSTM mode)
        'data_set': 'cat',  # datasets to train on
        'epochs': 50,  # how many times to go over the full train set (on average, since batches are drawn randomly)
        'save_every': None, # Batches between checkpoints creation and validation set evaluation. Once an epoch if None.
        'batch_size': 100,  # Minibatch size. Recommend leaving at 100.
        'accelerate_LSTM': False,  # Flag for using CuDNNLSTM layer, gpu + tf backend only
        # Loss Params:
        'optimizer': 'adam',  # adam or sgd
        'learning_rate': 0.001,
        'decay_rate': 0.9999,  # Learning rate decay per minibatch.
        'min_learning_rate': .00001,  # Minimum learning rate.
        'kl_tolerance': 0.2,  # Level of KL loss at which to stop optimizing for KL.
        'kl_weight': 0.5,  # KL weight of loss equation. Recommend 0.5 or 1.0.
        'kl_weight_start': 0.01,  # KL start weight when annealing.
        'kl_decay_rate': 0.99995,  # KL annealing decay rate per minibatch.
        'grad_clip': 1.0,  # Gradient clipping. Recommend leaving at 1.0.
        # Architecture Params:
        'z_size': 128,  # Size of latent vector z. Recommended 32, 64 or 128.
        'enc_rnn_size': 256,  # Units in encoder RNN.
        'dec_rnn_size': 512,  # Units in decoder RNN.
        'use_recurrent_dropout': True,  # Dropout with memory loss. Recommended
        'recurrent_dropout_prob': 0.9,  # Probability of recurrent dropout keep.
        'num_mixture': 20,  # Number of mixtures in Gaussian mixture model.
        # Data pre-processing Params:
        'random_scale_factor': 0.15,  # Random scaling data augmentation proportion.
        'augment_stroke_prob': 0.10  # Point dropping augmentation proportion.
    }

    return params_dict



    
class Seq2seqModel(object):

    def __init__(self, hps):
        # Hyper parameters
        self.hps = hps
        # Model
        self.model = self.build_model()
        # Print a model summary
        self.model.summary()

        # Optimizer
        if self.hps['optimizer'] == 'adam':
            self.optimizer = Adam(lr=self.hps['learning_rate'], clipvalue=self.hps['grad_clip'])
        elif self.hps['optimizer'] == 'sgd':
            self.optimizer = SGD(lr=self.hps['learning_rate'], momentum=0.9, clipvalue=self.hps['grad_clip'])
        else:
            raise ValueError('Unsupported Optimizer!')
        # Loss Function
        self.loss_func = self.model_loss()
        # Sample models, to be used when encoding\decoding specific strokes
        self.sample_models = {}

    def build_model(self):
        """ Create a Keras seq2seq VAE model for sketch-rnn """

        # Arranging inputs:
        self.encoder_input = Input(shape=(self.hps['max_seq_len'], 5), name='encoder_input')
        decoder_input = Input(shape=(self.hps['max_seq_len'], 5), name='decoder_input')

        # Set recurrent dropout to fraction of units to *drop*, if in CuDNN accelerated mode, don't use dropout:
        recurrent_dropout = 1.0-self.hps['recurrent_dropout_prob'] if \
            (self.hps['use_recurrent_dropout'] and (self.hps['accelerate_LSTM'] is False)) else 0

        # Option to use the accelerated version of LSTM, CuDNN LSTM. Much faster, but no support for recurrent dropout:
        if self.hps['accelerate_LSTM'] and self.hps['is_training']:
            lstm_layer_encoder = CuDNNLSTM(units=self.hps['enc_rnn_size'])
            lstm_layer_decoder = CuDNNLSTM(units=self.hps['dec_rnn_size'], return_sequences=True, return_state=True)
            self.hps['use_recurrent_dropout'] = False
            print('Using CuDNNLSTM - No Recurrent Dropout!')
        else:
            # Note that in inference LSTM is always selected (even in accelerated mode) so inference on CPU is supported
            lstm_layer_encoder = LSTM(units=self.hps['enc_rnn_size'], recurrent_dropout=recurrent_dropout)
            lstm_layer_decoder = LSTM(units=self.hps['dec_rnn_size'], recurrent_dropout=recurrent_dropout,
                                      return_sequences=True, return_state=True)

        # Encoder, bidirectional LSTM:
        encoder = Bidirectional(lstm_layer_encoder, merge_mode='concat')(self.encoder_input)

        # Latent vector - [batch_size]X[z_size]:
        self.batch_z = self.latent_z(encoder)

        # Decoder LSTM:
        self.decoder = lstm_layer_decoder

        # Initial state for decoder:
        self.initial_state = Dense(units=2*self.decoder.units, activation='tanh', name='dec_initial_state',
                              kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))
        initial_state = self.initial_state(self.batch_z)

        # Split to hidden state and cell state:
        init_h, init_c = (initial_state[:, :self.decoder.units], initial_state[:, self.decoder.units:])

        # Concatenate latent vector to expected output and  feed this as input to decoder:
        tile_z = RepeatVector(self.hps['max_seq_len'])(self.batch_z)
        decoder_full_input = Concatenate()([decoder_input, tile_z])

        # Retrieve decoder output tensors:
        [decoder_output, final_state1, final_state_2] = self.decoder(decoder_full_input, initial_state=[init_h, init_c])
        # self.final_state = [final_state1, final_state_2] todo: not used, remove when stable

        # Number of outputs:
        # 3 pen state logits, 6 outputs per mixture model(mean_x, mean_y, std_x, std_y, corr_xy, mixture weight pi)
        n_out = (3 + self.hps['num_mixture'] * 6)

        # Output FC layer
        self.output = Dense(n_out, name='output')
        output = self.output(decoder_output)

        # Build Keras model
        model_o = Model([self.encoder_input, decoder_input], output)

        return model_o

    def latent_z(self, encoder_output):
        """ Return a latent vector z of size [batch_size]X[z_size] """

        def transform2layer(z_params):
            """ Auxiliary function to feed into Lambda layer.
             Gets a list of [mu, sigma] and returns a random tensor from the corresponding normal distribution """
            mu, sigma = z_params
            sigma_exp = K.exp(sigma / 2.0)
            colored_noise = mu + sigma_exp*K.random_normal(shape=K.shape(sigma_exp), mean=0.0, stddev=1.0)
            return colored_noise
        # Dense layers to create the mean and stddev of the latent vector
        self.mu = Dense(units=self.hps['z_size'], kernel_initializer=RandomNormal(stddev=0.001))(encoder_output)
        self.sigma = Dense(units=self.hps['z_size'], kernel_initializer=RandomNormal(stddev=0.001))(encoder_output)

        # We cannot simply use the operations and feed to the next layer, so a Lambda layer must be used
        return Lambda(transform2layer)([self.mu, self.sigma])