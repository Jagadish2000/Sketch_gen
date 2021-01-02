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

        # Arrange inputs:
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

    def calculate_kl_loss(self, *args, **kwargs):
        #This function calculates the Kullback Leiber loss(KL loss)
        #KL loss= loss between the encoded svg input and the same decoded svg image

        kl_cost = -0.5*K.mean(1+self.sigma-K.square(self.mu)-K.exp(self.sigma))

        #Optimization of KL Loss will stop when kl_cost is lower than 'kl_tolerance'(hyperparameter)
        return K.maximum(kl_cost, self.hps['kl_tolerance'])

    def calculate_md_loss(self, y_true, y_pred):
        #This function calculates the Reconstruction loss 
        # Parse the output tensor to appropriate mixture density coefficients:
        out = self.get_mixture_coef(y_pred)
        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = out

        # Parse target vector to coordinates and pen states:
        [x1_data, x2_data] = [y_true[:, :, 0], y_true[:, :, 1]]
        pen_data = y_true[:, :, 2:5]

        # Get the density value of each mixture, estimated for the target coordinates:
        pdf_values = self.keras_2d_normal(x1_data, x2_data, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr)

        # Compute the GMM values (weighted sum of mixtures using pi values)
        gmm_values = pdf_values * o_pi
        gmm_values = K.sum(gmm_values, 2, keepdims=True)

        # gmm_loss is the loss wrt pen offset (L_s in equation 9 of https://arxiv.org/pdf/1704.03477.pdf)
        epsilon = 1e-6
        gmm_loss = -K.log(gmm_values + epsilon)  # avoid log(0)

        # Zero out loss terms beyond N_s, the last actual stroke
        fs = 1.0 - pen_data[:, :, 2]
        fs = K.expand_dims(fs)
        gmm_loss = gmm_loss * fs

        # pen_loss is the loss wrt pen state, (L_p in equation 9)
        pen_loss = categorical_crossentropy(pen_data, o_pen)
        pen_loss = K.expand_dims(pen_loss)

        # Eval mode, mask eos columns. todo: remove this?
        pen_loss = K.switch(K.learning_phase(), pen_loss, pen_loss * fs)

        # Total loss
        result = gmm_loss + pen_loss

        r_cost = K.mean(result)  # todo: Keras already averages over all tensor values, this might be redundant
        return r_cost


    def model_loss(self):
        #This function calculates the weight required to compute the total loss.
        #It also returns a function which calculates the complete loss.

        # KL loss
        kl_loss = self.calculate_kl_loss
        # Reconstruction loss
        md_loss_func = self.calculate_md_loss

        # Total loss= Reconstruction loss+ (KL weight)*(KL Loss)
        # KL weight is required for the total loss
        self.kl_weight = K.variable(self.hps['kl_weight_start'], name='kl_weight')
        kl_weight = self.kl_weight

        def seq2seq_loss(y_true, y_pred):
            # Function for final loss calculation to be passed to optimizer
            # Reconstruction loss
            md_loss = md_loss_func(y_true, y_pred)
            # Full loss
            model_loss = kl_weight*kl_loss() + md_loss
            return model_loss

        return seq2seq_loss



    def get_mixture_coef(self, out_tensor):
        """ Parses the output tensor to appropriate mixture density coefficients"""
        # This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.

        # Pen states:
        z_pen_logits = out_tensor[:, :, 0:3]
        # Process outputs into MDN parameters
        M = self.hps['num_mixture']
        dist_params = [out_tensor[:, :, (3 + M * (n - 1)):(3 + M * n)] for n in range(1, 7)]
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = dist_params

        # Softmax all the pi's and pen states:
        z_pi = softmax(z_pi)
        z_pen = softmax(z_pen_logits)

        # Exponent the sigmas and also make corr between -1 and 1.
        z_sigma1 = exponential(z_sigma1)
        z_sigma2 = exponential(z_sigma2)
        z_corr = tanh(z_corr)

        r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits]
        return r

    def keras_2d_normal(self, x1, x2, mu1, mu2, s1, s2, rho):
        """ Returns the density values of each mixture, estimated for the target coordinates tensors
        This is the result of eq # 24 of http://arxiv.org/abs/1308.0850."""
        M = mu1.shape[2]  # Number of mixtures
        norm1 = K.tile(K.expand_dims(x1), [1, 1, M]) - mu1
        norm2 = K.tile(K.expand_dims(x2), [1, 1, M]) - mu2
        s1s2 = s1 * s2
        # eq 25
        z = K.square(norm1 / s1) + K.square(norm2 / s2) - 2.0 * (rho * norm1 * norm2) / s1s2
        neg_rho = 1.0 - K.square(rho)
        result = K.exp((-z) / (2 * neg_rho))
        denom = 2 * np.pi * s1s2 * K.sqrt(neg_rho)
        result = result / denom
        return result

    def compile(self):
        """ Compiles the Keras model. Includes metrics to differentiate between the two main loss terms """
        self.model.compile(optimizer=self.optimizer, loss=self.loss_func,
                           metrics=[self.calculate_md_loss, self.calculate_kl_loss])
        print('Model Compiled!')

    def load_trained_weights(self, weights):
        """ Loads weights of a pre-trained model. 'weights' is path to h5 model \ weights file"""
        self.model.load_weights(weights)
        print('Weights from {} loaded successfully'.format(weights))
