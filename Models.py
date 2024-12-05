import tensorflow as tf
import time

# 즉시 실행 모드 활성화
# tf.config.run_functions_eagerly(True)
tf.config.run_functions_eagerly(False)

class WGAN_GP:
    def __init__(self, latent_dim, data_dim, batch_size, num_epochs, learning_rate, n_critic=5, label_classes=2):
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.n_critic = n_critic
        self.label_classes = label_classes
        self.test = 'test'
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)

    def build_generator(self):
        noise = tf.keras.Input(shape=(self.latent_dim,), name='Noise')
        labels = tf.keras.Input(shape=(1,), dtype="int32", name='Labels')

        # 레이블 임베딩 및 결합
        label_embedding = tf.keras.layers.Embedding(self.label_classes, self.latent_dim, name="Embedding")(labels)
        label_embedding = tf.keras.layers.Flatten(name="Flatten")(label_embedding)
        model_input = tf.keras.layers.Concatenate(name="Concatenate")([noise, label_embedding])

        x = tf.keras.layers.Dense(256)(model_input)
        x = tf.keras.layers.BatchNormalization()(x)  # Batch Normalization 추가
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(512)(x)
        # x = tf.keras.layers.BatchNormalization()(x)  # Batch Normalization 추가
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(1024)(x)
        # x = tf.keras.layers.BatchNormalization()(x)  # Batch Normalization 추가
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        #output = tf.keras.layers.Dense(self.data_dim, activation="tanh")(x)
        output = tf.keras.layers.Dense(self.data_dim)(x)

        model = tf.keras.Model(inputs=[noise, labels], outputs=output, name="Generator")
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.data_dim,)),
            tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1)
        ], name="Discriminator")
        return model

    def wasserstein_loss(self, y_true, y_pred):
        return tf.keras.backend.mean(y_true * y_pred)

    def gradient_penalty(self, real_data, fake_data):
        # alpha = tf.random.uniform((self.batch_size, 1))
        real_data = tf.cast(real_data, dtype=tf.float32)  # float32로 변환
        fake_data = tf.cast(fake_data, dtype=tf.float32)  # float32로 변환
        alpha = tf.random.uniform([tf.shape(real_data)[0], 1], 0.0, 1.0)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            d_interpolated = self.discriminator(interpolated)
        gradients = tape.gradient(d_interpolated, [interpolated])[0]
        gradients_sqr = tf.square(gradients)
        gradients_norm = tf.sqrt(tf.reduce_sum(gradients_sqr, axis=[1]))
        gradient_penalty = tf.reduce_mean((gradients_norm - 1.0) ** 2)
        return gradient_penalty

    def train_step(self, real_data, labels, train_generator=True):
        batch_size = tf.shape(real_data)[0]

        # 라벨 데이터를 int32로 변환
        labels = tf.cast(labels, tf.int32)

        # 가짜 데이터 생성
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        generated_data = self.generator([random_latent_vectors, labels])

        if not train_generator:
            # Discriminator 학습 (진짜와 가짜 데이터 구분)
            with tf.GradientTape() as disc_tape:
                disc_tape.watch(real_data)
                real_output = self.discriminator(real_data)
                fake_output = self.discriminator(generated_data)
                d_loss_real = self.wasserstein_loss(tf.ones_like(real_output), real_output)
                d_loss_fake = self.wasserstein_loss(-tf.ones_like(fake_output), fake_output)
                gp = self.gradient_penalty(real_data, generated_data)
                lambda_gp = 5.0
                d_loss = d_loss_real + d_loss_fake + lambda_gp * gp

            gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))
            return d_loss, None
        else:
            # Generator 학습 (Discriminator를 속이도록 가짜 데이터를 생성)
            with tf.GradientTape() as gen_tape:
                generated_data = self.generator([random_latent_vectors, labels])
                self.test = generated_data
                fake_output = self.discriminator(generated_data)
                g_loss = self.wasserstein_loss(tf.ones_like(fake_output), fake_output)

            gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            return None, g_loss

    def train(self, dataset):
        start_time = time.time()
        before_epoch_time = time.time()
        for epoch in range(self.num_epochs):
            total_batches = len(dataset)
            for i, (real_data, labels) in enumerate(dataset):

                if labels.dtype != tf.int32:
                    labels = tf.strings.to_number(labels, out_type=tf.int32)
                d_loss = 0
                # n_critic 만큼 Discriminator 학습
                for _ in range(self.n_critic):
                    d_loss, _ = self.train_step(real_data, labels, train_generator=False)
                # Generator 학습
                _, g_loss = self.train_step(real_data, labels, train_generator=True)

                # Epoch 마다 손실 출력
                if i % 100 == 0:
                    print(
                        f'Epoch [{epoch + 1}/{self.num_epochs}] - Batch {i}/{total_batches}, d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}')
            epoch_time = time.time() - before_epoch_time
            print(f'{epoch} Epoch time: {epoch_time:.4f}')
            before_epoch_time = time.time()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training Completed. Total Training Time: {elapsed_time:.2f} seconds")


class VanillaGAN:
    def __init__(self, latent_dim, data_dim, batch_size, num_epochs, learning_rate, label_classes=2):
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.label_classes = label_classes

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)

    def build_generator(self):
        noise = tf.keras.Input(shape=(self.latent_dim,), name='Noise')
        labels = tf.keras.Input(shape=(1,), dtype="int32", name='Labels')

        # 레이블 임베딩 및 결합
        label_embedding = tf.keras.layers.Embedding(self.label_classes, self.latent_dim, name="Embedding")(labels)
        label_embedding = tf.keras.layers.Flatten(name="Flatten")(label_embedding)
        model_input = tf.keras.layers.Concatenate(name="Concatenate")([noise, label_embedding])

        x = tf.keras.layers.Dense(256)(model_input)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dense(512)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        output = tf.keras.layers.Dense(self.data_dim, activation="tanh")(x)

        model = tf.keras.Model(inputs=[noise, labels], outputs=output, name="Generator")
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.data_dim,)),
            tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
            tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
            tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ], name="Discriminator")
        return model

    def gan_loss(self, y_true, y_pred):
        return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

    def train_step(self, real_data, labels, train_generator=True):
        batch_size = tf.shape(real_data)[0]

        # 가짜 데이터 생성
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        labels = tf.cast(labels, dtype=tf.int32)  # labels를 int32로 변환

        if not train_generator:
            # Discriminator 학습 (진짜와 가짜 데이터 구분)
            with tf.GradientTape() as disc_tape:
                disc_tape.watch(self.discriminator.trainable_variables)
                real_data = tf.cast(real_data, dtype=tf.float32)  # real_data를 Tensor로 변환 (float32)
                generated_data = self.generator([random_latent_vectors, labels], training=True)

                real_output = self.discriminator(real_data, training=True)
                fake_output = self.discriminator(generated_data, training=True)

                d_loss_real = self.gan_loss(tf.ones_like(real_output), real_output)
                d_loss_fake = self.gan_loss(tf.zeros_like(fake_output), fake_output)
                d_loss = d_loss_real + d_loss_fake

            gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
            if None in gradients_of_discriminator:
                print("Warning: Some gradients are None. Check the computation graph.")
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))
            return d_loss, None
        else:
            # Generator 학습 (Discriminator를 속이도록 가짜 데이터를 생성)
            with tf.GradientTape() as gen_tape:
                gen_tape.watch(self.generator.trainable_variables)
                generated_data = self.generator([random_latent_vectors, labels], training=True)
                fake_output = self.discriminator(generated_data, training=True)
                g_loss = self.gan_loss(tf.ones_like(fake_output), fake_output)

            gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
            if None in gradients_of_generator:
                print("Warning: Some gradients are None. Check the computation graph.")
            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            return None, g_loss

    def train(self, dataset):
        start_time = time.time()
        before_epoch_time = time.time()
        for epoch in range(self.num_epochs):
            total_batches = len(dataset)
            for i, (real_data, labels) in enumerate(dataset):
                d_loss = 0
                # Discriminator 학습
                d_loss, _ = self.train_step(real_data, labels, train_generator=False)
                # Generator 학습
                _, g_loss = self.train_step(real_data, labels, train_generator=True)

                # Epoch 마다 손실 출력
                if i % 100 == 0:
                    print(
                        f'Epoch [{epoch + 1}/{self.num_epochs}] - Batch {i}/{total_batches}, d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}')
            epoch_time = time.time() - before_epoch_time
            print(f'{epoch} Epoch time: {epoch_time:.4f}')
            before_epoch_time = time.time()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Training Completed. Total Training Time: {elapsed_time:.2f} seconds")