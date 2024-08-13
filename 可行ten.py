import tensorflow as tf
import os
import time

# 数据集路径
dataset_path = 'C:/Users/admin/Desktop/AI风景model升级版/dataset'
BUFFER_SIZE = 400
BATCH_SIZE = 16  # 增加 batch size
IMG_WIDTH = 256
IMG_HEIGHT = 256

# 加载并预处理数据
def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)  # 确保解码为 3 通道
    image = tf.cast(image, tf.float32)
    return image

def resize(image, height, width):
    image = tf.image.resize(image, [height, width])
    return image

def normalize(image):
    image = (image / 127.5) - 1
    return image

def load_image_train(blurry_file, clear_file):
    blurry_image = load(blurry_file)
    clear_image = load(clear_file)
    blurry_image = resize(blurry_image, IMG_HEIGHT, IMG_WIDTH)
    clear_image = resize(clear_image, IMG_HEIGHT, IMG_WIDTH)
    blurry_image = normalize(blurry_image)
    clear_image = normalize(clear_image)
    return blurry_image, clear_image

def load_image_test(blurry_file, clear_file):
    blurry_image = load(blurry_file)
    clear_image = load(clear_file)
    blurry_image = resize(blurry_image, IMG_HEIGHT, IMG_WIDTH)
    clear_image = resize(clear_image, IMG_HEIGHT, IMG_WIDTH)
    blurry_image = normalize(blurry_image)
    clear_image = normalize(clear_image)
    return blurry_image, clear_image

def get_dataset_paths(dataset_path, phase):
    blurry_images = sorted(os.listdir(os.path.join(dataset_path, phase, 'blurry')))
    clear_images = sorted(os.listdir(os.path.join(dataset_path, phase, 'clear')))
    blurry_paths = [os.path.join(dataset_path, phase, 'blurry', img) for img in blurry_images]
    clear_paths = [os.path.join(dataset_path, phase, 'clear', img) for img in clear_images]
    return blurry_paths, clear_paths

blurry_train_paths, clear_train_paths = get_dataset_paths(dataset_path, 'train')
blurry_test_paths, clear_test_paths = get_dataset_paths(dataset_path, 'test')

def parse_image(blurry_file, clear_file, is_training=True):
    if is_training:
        blurry_image, clear_image = load_image_train(blurry_file, clear_file)
    else:
        blurry_image, clear_image = load_image_test(blurry_file, clear_file)

    return blurry_image, clear_image

train_dataset = tf.data.Dataset.from_tensor_slices((blurry_train_paths, clear_train_paths))
train_dataset = train_dataset.map(
    lambda blurry, clear: tf.py_function(parse_image, [blurry, clear, True], [tf.float32, tf.float32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((blurry_test_paths, clear_test_paths))
test_dataset = test_dataset.map(
    lambda blurry, clear: tf.py_function(parse_image, [blurry, clear, False], [tf.float32, tf.float32]))
test_dataset = test_dataset.batch(BATCH_SIZE)

# 构建Pix2Pix模型
OUTPUT_CHANNELS = 3

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                               kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same',
                                           kernel_initializer=initializer, activation='tanh')
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])
    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_loss, gan_loss, l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def compute_metrics(dataset):
    total_gen_loss = 0.0
    total_disc_loss = 0.0
    num_batches = 0

    for input_image, target in dataset:
        gen_loss, disc_loss = train_step(input_image, target, 0)
        total_gen_loss += gen_loss
        total_disc_loss += disc_loss
        num_batches += 1

    return total_gen_loss / num_batches, total_disc_loss / num_batches

def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()
        print(f"Starting Epoch {epoch + 1}/{epochs}")

        for n, (input_image, target) in train_ds.enumerate():
            gen_loss, disc_loss = train_step(input_image, target, epoch)

            if (n + 1) % 100 == 0:  # 每 100 次 batch 打印一次损失
                print(
                    f"Epoch {epoch + 1} Batch {n + 1} Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}")

        # 计算训练和测试集的损失
        train_gen_loss, train_disc_loss = compute_metrics(train_ds)
        test_gen_loss, test_disc_loss = compute_metrics(test_ds)

        print(f"Epoch {epoch + 1} Training Gen Loss: {train_gen_loss:.4f}, Disc Loss: {train_disc_loss:.4f}")
        print(f"Epoch {epoch + 1} Test Gen Loss: {test_gen_loss:.4f}, Disc Loss: {test_disc_loss:.4f}")

        print(f"Completed Epoch {epoch + 1} in {time.time() - start} seconds")

    # 保存训练好的模型
    generator.save('C:/Users/admin/Desktop/AI风景model升级版/model-gen/generator2.h5')
    discriminator.save('C:/Users/admin/Desktop/AI风景model升级版/model-dis/discriminator2.h5')

# 训练模型
fit(train_dataset, epochs=10, test_ds=test_dataset)
