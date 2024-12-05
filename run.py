#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()
import warnings

import pandas as pd
import sys
import os
import tensorflow as tf
import math

from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding, multiply, LeakyReLU, ReLU, Softmax, Lambda, Layer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler, Normalizer
import numpy as np
from lib.predictions import preproc_data

from tensorflow.python.ops.numpy_ops import np_config

# custom library import
from lib.predictions import calculate_fid


# 전역변수 선언 및 초기 설정
class GlobalInitial:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        global G_DLOSS, G_GLOSS, G_RLOSS, G_CLOSS, G_FLOSS, G_CACC
        np_config.enable_numpy_behavior()
        script_dir = os.path.dirname(os.path.abspath("lib"))
        sys.path.append(os.path.dirname(script_dir))

        warnings.simplefilter("ignore")
        try:
            # 사용 가능한 GPU 확인
            print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print("GPU is available. Using the following GPU devices:")
                for gpu in gpus:
                    print(gpu)
            else:
                print("No GPU devices available. Using CPU instead.")
        except RuntimeError as e:
            print(e)



class GradientPenaltyLayer(Layer):
    def __init__(self, critic, weight=10.0, **kwargs):
        super().__init__(**kwargs)
        self.critic = critic
        self.weight = weight

    def call(self, inputs):
        averaged_samples, label = inputs
        with tf.GradientTape() as tape:
            tape.watch(averaged_samples)
            validity = self.critic([averaged_samples, label])
        gradients = tape.gradient(validity, averaged_samples)
        gradients_sqr = tf.square(gradients)
        gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=tf.range(1, tf.rank(gradients_sqr)))
        gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
        gradient_penalty = tf.square(1 - gradient_l2_norm)
        mean_gradient_penalty = tf.reduce_mean(gradient_penalty)
        #tf.print("Gradient Penalty:", mean_gradient_penalty)
        self.add_loss(self.weight * mean_gradient_penalty)
        return validity


# Model Definition
class RandomWeightedAverage(tf.keras.layers.Layer):
    """Provides a (random) weighted average between real and generated image samples"""

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        alpha = tf.random.uniform((self.batch_size, 1), 0.0, 1.0)
        ev = (alpha * inputs[0]) + ((1 - alpha) * inputs[1])
        return tf.convert_to_tensor(ev)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


# EC-GAN Class
class ECGAN():

    def __init__(self,
                 x_train,
                 y_train,
                 num_classes: int,
                 latent_dim: int,
                 batch_size: int,
                 n_critic: int,
                 conf_thresh: float,
                 adv_weight: float):
        """Implement EC-GAN with an WCGAN-GP and MLP.

        Attributes
        ---------
        x_train : numpy.ndarray
            Real data without labels used for training.
            (Created with sklearn.model_selection.train_test_split

        y_train : numpy.ndarray
            Real data labels.

        num_classes : int
            Number of data classes. Number of unique elements in y_train.

        data_dim : int
            Data dimension. Number of columns in x_train.

        latent_dim : int
            Dimension of random noise vector (z), used for training
            the generator.

        batch_size : int
            Size of training batch in each epoch.

        n_critic : int
            Number of times the critic (discriminator) will be trained
            in each epoch.

        conf_thresh : float
            Confidence threshold. EC-GAN parameter which decides how good
            the generated sample needs to be, for it to be fed to the
            classifier.

        adv_weight : float
            Adverserial weight. EC-GAN parameter which represents the
            importance fake data has on classifier training.
            Value has been taken from the original paper.

        """

        self.x_train = x_train.copy()
        self.y_train = y_train.copy()

        # Store labels as one-hot vectors.
        self.y_train_onehot = to_categorical(y_train)

        self.num_classes = num_classes
        # py3.8.9
        self.data_dim = x_train.shape[1]
        #self.data_dim = np.zeros((x_train.shape[1],))

        self.latent_dim = latent_dim
        self.batch_size = batch_size

        # WCGAN-GP parameters.
        self.n_critic = n_critic

        # EC-GAN parameters.
        self.conf_thresh = conf_thresh
        self.adv_weight = adv_weight

        # Log training progress.
        self.losslog = []
        self.class_acc_log = []
        self.class_loss_log = []

        # Adam optimizer for WCGAN-GP, suggested by original paper.
        #optimizer = Adam(learning_rate=0.0005, beta_1=0.05, beta_2=0.9)


        # Categorical crossentropy loss function for the classifier.
        self.cce_loss = tf.keras.losses.CategoricalCrossentropy()

        # Build the generator, critic and classifier
        self.generator = self.build_generator()
        self.critic = self.build_critic()
        self.classifier = self.build_classifier()
        self.wasserstein_loss = self.wasserstein_loss_function()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.05, beta_2=0.9)

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic.
        self.generator.trainable = False

        # 데이터 입력 정의
        real_data = Input(shape=(self.data_dim,), name="Real_data")
        noise = Input(shape=(self.latent_dim,), name="Noise")
        label = Input(shape=(1,), dtype="int32", name="Label")

        # Generator를 통해 fake_data 생성
        fake_data = self.generator([noise, label])

        # real_data와 fake_data 사이의 랜덤 가중 평균 생성

        self.compile_critic(self.generator)
        # self.critic_model.compile(loss=[self.wasserstein_loss,
        #                                 self.wasserstein_loss,
        #                                 #lambda y_true, y_pred: self.gradient_penalty_loss(y_true, y_pred, interpolated_data)
        #                                 #partial_gp_loss
        #                                 ],
        #                           optimizer=optimizer,
        #                           # run_eagerly=True,
        #                           loss_weights=[1, 1, 10])

        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze other's layers.
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator.
        noise = Input(shape=(self.latent_dim,), name="Noise")

        # Add label to input.
        label = Input(shape=(1,), name="Label")

        # Generate data based of noise.
        fake_data = self.generator([noise, label])

        # Discriminator determines validity.
        valid = self.critic([fake_data, label])

        # Defines generator model.
        self.generator_model = Model([noise, label], valid)

        self.generator_model.compile(loss=self.wasserstein_loss,
                                     optimizer=self.optimizer)



        #-------------------------------
        # Construct Computational Graph
        #   for the Classifier (real)
        #-------------------------------

        # Real data classifier training

        #real_data = Input(shape=self.data_dim, name="Real_data")
        real_data = Input(shape=(self.data_dim,), name="Real_data")

        real_predictions = self.classifier(real_data)

        self.real_classifier_model = Model(real_data, real_predictions)

        self.real_classifier_model.compile(loss="categorical_crossentropy",
                                           optimizer="adamax",
                                           metrics=["accuracy"])

        #-------------------------------
        # Construct Computational Graph
        #   for the Classifier (fake)
        #-------------------------------

        # Fake data classifier training

        noise = Input(shape=(self.latent_dim,), name="Noise")
        fake_labels = Input(shape=(1,), name="Label")

        #real_data = Input(shape=self.data_dim, name="Real_data")
        real_data = Input(shape=(self.data_dim,), name="Real_data")

        fake_data = self.generator([noise, fake_labels])

        fake_predictions = self.classifier(fake_data)

        self.fake_classifier_model = Model([noise, fake_labels], fake_predictions)

        self.fake_classifier_model.compile(loss=self.ecgan_loss,
                                           optimizer="adamax",
                                           metrics=["accuracy"])



    def ecgan_loss(self, y_true, y_pred):
        """Calculate loss for fake data predictions."""
        class lossLayer(tf.keras.layers.Layer):
            def call(self, inputs, fn, options=None):
                if options is None:
                    return fn(inputs)
                else:
                    return fn(inputs, options)

        # 사용 예시
        # layer = lossLayer()
        # max_values = layer(y_pred, tf.math.reduce_max, axis=1)
        max_values = tf.math.reduce_max(y_pred, axis=1)

        max_index = tf.where(tf.math.greater(max_values, self.conf_thresh))
        # max_index = layer(tf.math.greater(max_values, self.conf_thresh), tf.where)

        loss = self.adv_weight * self.cce_loss(y_true[max_index], y_pred[max_index])

        return loss


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """"
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        # print(y_pred, averaged_samples)
        def compute_gradient_penalty(inputs):
            averaged_samples, y_pred = inputs
            with tf.GradientTape() as tape:
                tape.watch(averaged_samples)
                validity = self.critic([averaged_samples, y_pred])
                gradients = tape.gradient(validity, averaged_samples)
                return gradients

        gradient_penalty_layer = Lambda(compute_gradient_penalty)
        gradients = gradient_penalty_layer([averaged_samples, y_pred])
        # compute the euclidean norm by squaring ...
        gradients_sqr = Lambda(lambda x: tf.square(x))(gradients)
        gradients_sqr_sum = Lambda(lambda x: tf.reduce_sum(x, axis=np.arange(1, len(x.shape))))(gradients_sqr)
        gradient_l2_norm = Lambda(lambda x: tf.sqrt(x))(gradients_sqr_sum)

        gradient_penalty = Lambda(lambda x: tf.square(1 - x))(gradient_l2_norm)
        # return the mean as loss over all the batch samples
        means = Lambda(lambda x: tf.reduce_mean(x))(gradient_penalty)
        #print(means)
        #print(K.eval(means))
        return means


    def wasserstein_loss_function(self):
        def wasserstein_loss(y_true, y_pred):
            return K.mean(y_true * y_pred)
        return wasserstein_loss


    def build_generator(self):

        #model = Sequential(name="Generator")

        # First hidden layer.
        # model.add(Dense(256, input_dim=self.latent_dim))
        # model.add(LeakyReLU(negative_slope=0.2))
        # model.add(Dropout(0.3))
        #
        # # Second hidden layer.
        # model.add(Dense(512))
        # model.add(LeakyReLU(negative_slope=0.2))
        # model.add(Dropout(0.3))
        #
        # # Third hidden layer.
        # model.add(Dense(1024))
        # model.add(LeakyReLU(negative_slope=0.2))
        # model.add(Dropout(0.3))
        #
        # # Output layer.
        # model.add(Dense(self.data_dim, activation="tanh"))
        #
        # model.summary()
        # Noise and label input layers.
        noise = Input(shape=(self.latent_dim,), name="Noise")
        label = Input(shape=(1,), dtype="int32", name="Label")

        # Embed labels into onehot encoded vectors.
        label_embedding = Flatten(name="Flatten")(Embedding(self.num_classes, self.latent_dim, name="Embedding")(label))

        # Multiply noise and embedded labels to be used as model input.
        model_input = multiply([noise, label_embedding], name="Multiply")

        x = Dense(256)(noise)
        x = LeakyReLU(negative_slope=0.2)(x)
        x = Dropout(0.3)(x)
        x = Dense(512)(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        x = Dropout(0.3)(x)
        x = Dense(1024)(x)
        x = LeakyReLU(negative_slope=0.2)(x)
        x = Dropout(0.3)(x)
        x = Dense(self.data_dim, activation="tanh")(x)

        model = Model(inputs=noise, outputs=x, name="Generator")



        generated_data = model(model_input)

        return Model(inputs=[noise, label],
                     outputs=generated_data,
                     name="Generator")

    def build_critic(self):
        model_input = Input(shape=(self.data_dim,), name="Critic_Input")
        label = Input(shape=(1,), dtype="int32", name="Label")
        x = Dense(1024)(model_input)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(256)(x)
        x = LeakyReLU(alpha=0.2)(x)
        validity = Dense(1)(x)
        return Model(inputs=[model_input, label], outputs=validity, name="Critic")


    def compile_critic(self, generator):
        self.generator = generator
        self.generator.trainable = False

        # 데이터 입력 정의
        real_data = Input(shape=(self.data_dim,), name="Real_data")
        noise = Input(shape=(self.latent_dim,), name="Noise")
        label = Input(shape=(1,), dtype="int32", name="Label")

        # Generator를 통해 fake_data 생성
        fake_data = self.generator([noise, label])

        # Critic을 통해 valid, fake 판별
        valid = self.critic([real_data, label])
        fake = self.critic([fake_data, label])

        # real_data와 fake_data 사이의 랜덤 가중 평균 생성
        interpolated_data = RandomWeightedAverage(self.batch_size)([real_data, fake_data])

        # Gradient Penalty 계산 및 모델 손실에 추가
        gradient_penalty = GradientPenaltyLayer(self.critic, weight=10.0)([interpolated_data, label])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.05, beta_2=0.9)

        # Critic 모델 구성: valid, fake, gradient_penalty
        self.critic_model = Model(
            inputs=[real_data, label, noise],
            outputs=[valid, fake, gradient_penalty]
        )

        # 모델 컴파일: gradient_penalty는 add_loss로 처리되므로 손실 리스트에서 제외
        self.critic_model.compile(
            loss=[
                self.wasserstein_loss,  # valid에 대한 손실
                self.wasserstein_loss,  # fake에 대한 손실
                None  # gradient_penalty는 손실 리스트에서 제외
            ],
            optimizer=optimizer,
            loss_weights=[1, 1, 0]  # gradient_penalty의 가중치는 0으로 설정
        )

    def build_classifier(self):

        # model = Sequential(name="Classifier")
        #
        # # First hidden layer.
        # model.add(Dense(128, input_dim=self.data_dim))
        # model.add(ReLU())
        # model.add(Dropout(0.3))
        #
        # # Second hidden layer.
        # model.add(Dense(256))
        # model.add(ReLU())
        # model.add(Dropout(0.3))
        #
        # model.add(Dense(128))
        # model.add(ReLU())
        # model.add(Dropout(0.3))
        #
        # # Output layer.
        # model.add(Dense(self.num_classes))
        # model.add(Softmax())
        #
        # model.summary()

        # Data input.
        #data = Input(shape=self.data_dim, name="Data")
        data = Input(shape=(self.data_dim,), name="Data")
        x = Dense(128)(data)
        x = ReLU()(x)
        x = Dropout(0.3)(x)
        x = Dense(256)(x)
        x = ReLU()(x)
        x = Dropout(0.3)(x)
        x = Dense(128)(x)
        x = ReLU()(x)
        x = Dropout(0.3)(x)

        # CLassifier outout is class predictions vector.
        # predictions = model(data)
        predictions = Dense(self.num_classes)(x)
        predictions = Softmax()(predictions)

        return Model(inputs=data,
                     outputs=predictions,
                     name="Classifier")


    def train(self, epochs):

        self.epochs = epochs

        # Adversarial ground truths.
        valid = -(np.ones((self.batch_size, 1)))
        #valid = -tf.ones((self.batch_size, 1))
        fake =  np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1))

        # Number of batches.
        self.n_batches = math.floor(self.x_train.shape[0] / self.batch_size)

        overhead = self.x_train.shape[0] % self.batch_size

        for epoch in range(epochs):

            # Reset training set.
            self.x_train = x_train.copy()
            self.y_train = y_train.copy()

            # Select random overhead rows that do not fit into batches.
            rand_overhead_idx = np.random.choice(range(self.x_train.shape[0]), overhead, replace=False)

            # Remove random overhead rows.
            self.x_train = np.delete(self.x_train, rand_overhead_idx, axis=0)
            self.y_train = np.delete(self.y_train, rand_overhead_idx, axis=0)


            # Split training data into batches.
            x_batches = np.split(self.x_train, self.n_batches)
            y_batches = np.split(self.y_train, self.n_batches)

            for x_batch, y_batch, i in zip(x_batches, y_batches, range(self.n_batches)):
                #print(f"i:{i}, epoch:{epoch}, N-batch:{range(self.n_batches)}")

                if epoch < 5:

                    for _ in range(self.n_critic):

                        # ---------------------
                        #  Train Critic
                        # ---------------------

                        # Generate random noise.
                        noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
                        # 기존 NumPy 방식에서 수정된 TensorFlow 방식

                        # Train the critic.
                        #3.8.9
                        d_loss = self.critic_model.train_on_batch([x_batch, y_batch, noise], [valid, fake, dummy])
                        #print(f"Epoch {epoch}, Loss: {d_loss}")

                    # ---------------------
                    #  Train Generator
                    # ---------------------

                    # Generate sample of artificial labels.
                    noise = tf.random.normal(shape=(self.batch_size, self.latent_dim))

                    #generated_labels = np.random.randint(1, self.num_classes, self.batch_size).reshape(-1, 1)
                    generated_labels = tf.random.uniform(
                        shape=(self.batch_size, 1),
                        minval=1,
                        maxval=self.num_classes,
                        dtype=tf.int32
                    )

                    # Train generator.
                    g_loss = self.generator_model.train_on_batch([noise, generated_labels], valid)


                    # ---------------------
                    #  Train Classifier
                    # ---------------------

                    # One-hot encode real labels.
                    y_batch = to_categorical(y_batch, self.num_classes)

                    # One-hot encode generated labels.
                    generated_labels_onehot = to_categorical(generated_labels, self.num_classes)

                    real_loss = self.real_classifier_model.train_on_batch(x_batch, y_batch)

                    fake_loss = self.fake_classifier_model.train_on_batch([noise, generated_labels], generated_labels_onehot)

                    # Classifier loss as presented in EC-GAN paper.
                    c_loss = (real_loss[0] + fake_loss[0]) / (1 + self.adv_weight)

                    avg_acc = np.mean([real_loss[1], fake_loss[1]])

                else:

                    # ---------------------
                    #  Train Classifier
                    # ---------------------

                    # Generate random noise.
                    noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))

                    # Generate sample of artificial labels.
                    generated_labels = np.random.randint(1, self.num_classes, self.batch_size).reshape(-1, 1)

                    # One-hot encode real labels.
                    y_batch = to_categorical(y_batch, self.num_classes)

                    # One-hot encode generated labels.
                    generated_labels_onehot = to_categorical(generated_labels, self.num_classes)

                    real_loss = self.real_classifier_model.train_on_batch(x_batch, y_batch)

                    fake_loss = self.fake_classifier_model.train_on_batch([noise, generated_labels], generated_labels_onehot)

                    # Classifier loss as presented in EC-GAN paper.
                    c_loss = (real_loss[0] + fake_loss[0]) / (1 + self.adv_weight)

                    avg_acc = np.mean([real_loss[1], fake_loss[1]])


                # ---------------------
                #  Logging
                # ---------------------

                # self.losslog.append([d_loss[0], g_loss, c_loss])
                self.class_loss_log.append([real_loss[0], fake_loss[0], c_loss])
                self.class_acc_log.append([real_loss[1], fake_loss[1], avg_acc])

                # Plot progress.
                DLOSS = "%.4f" % d_loss[0]
                GLOSS = "%.4f" % g_loss
                CLOSS = "%.4f" % c_loss
                RLOSS = "%.4f" % real_loss[0]
                FLOSS = "%.4f" % fake_loss[0]
                CACC  = "%.4f" % real_loss[1]

                if i % 100 == 0:
                    print (f"{epoch} - {i}/{self.n_batches} \t [D loss: {DLOSS}] [G loss: {GLOSS}] [R loss: {RLOSS} | F loss: {FLOSS} | C loss: {CLOSS} - C acc: {CACC}]")

        global G_DLOSS, G_GLOSS, G_RLOSS, G_CLOSS, G_FLOSS, G_CACC
        G_DLOSS = "%.4f" % d_loss[0]
        G_GLOSS = "%.4f" % g_loss
        G_CLOSS = "%.4f" % c_loss
        G_RLOSS = "%.4f" % real_loss[0]
        G_FLOSS = "%.4f" % fake_loss[0]
        G_CACC = "%.4f" % real_loss[1]




if __name__ == '__main__':
    GlobalInitial()

    # Read Data
    train = pd.read_csv("./data/Train.csv", low_memory=False)
    test = pd.read_csv("./data/Test.csv", low_memory=False)

    data = pd.concat([train, test], ignore_index=True)

    x_train, x_test, y_train, y_test = preproc_data(data, train_sample=0.7, pca_dim=31)

    #GAN 모델 학습 (라벨 0, 1)
    gan = ECGAN(x_train,
                y_train,
                num_classes=2,
                latent_dim=31,
                batch_size=128,
                n_critic=5,
                conf_thresh=.2,
                adv_weight=.1
                )

    gan.train(epochs=10)

    # 임의의 노이즈 생성 (latent_dim 크기의 랜덤 벡터)
    noise = np.random.normal(0, 1, (x_test.shape[0], 31))  # latent_dim = 32로 설정

    # 임의의 라벨 생성 (0 또는 1)
    random_labels = np.random.randint(0, 2, (x_test.shape[0], 1))

    # 로드한 생성기를 통해 가짜 데이터 생성
    generated_data = gan.generator.predict([noise, random_labels])
    #gen_data = gan.generator([noise, random_labels])
    # 데이터 정규화 (0과 1 사이로 변환)
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 실제 데이터 정규화
    x_test_scaled = scaler.fit_transform(x_test)

    generated_data_scaled = scaler.transform(generated_data)
    norm = Normalizer().fit(x_test_scaled)

    x_test_scaled = norm.transform(x_test_scaled)
    generated_data_scaled = norm.transform(generated_data_scaled)

    # FID 계산
    real_data = x_test_scaled
    fid_score = calculate_fid(generated_data_scaled, np.array(real_data))
    print(f"FID Score: {fid_score}")

    file_path = './result/'  # 파일을 생성할 경로와 파일 이름을 지정
    file_content = f"FID Score: {fid_score}\n"
    file_nm = f"FID_score"
    up_file = file_path + file_nm

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    with open(f"{up_file}.txt", 'w') as f:
        f.write(file_content)
        #    print(f"Created File: {up_file}{num}.txt")
        print(f"Created File: {up_file}.txt")
        print(f"End Of {sys.argv[0]}")


