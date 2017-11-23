"""Routine for decoding the mnist binary file format."""
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 50,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the umd data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

height = 28
width = 28
depth = 1

def read_mnist(filename_queue):
  class ImageRecord(object):
    pass
  result = ImageRecord()

  # Dimensions of the images in the dataset.
  label_bytes = 1
  # Set the following constants as appropriate.
  result.height = 28
  result.width = 28
  result.depth = 1
  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + image_bytes

  assert record_bytes == 785  # Based on your question.

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the binary, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                           [result.depth, result.height, result.width])
  # Convert from [depth, height, width] to [height, width, depth].
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle, numPrThread=16):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 1] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = numPrThread
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)
        
  return images, tf.reshape(label_batch, [batch_size])

def _distorted_inputs(batch_size, randFlip=True):
  """Construct distorted input for umd training using the Reader ops.

  Args:
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """  
  # filenames = ["/raid5/hasnat/tf/vmf/dataset/mnist_train.bin"]
  filenames = ["/home/hasnat/Desktop/mnist_verify/vae_mnist/mnist_train.bin"]

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_mnist(filename_queue)
  
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.scalar_mul(0.0039, reshaped_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.7
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d MNIST images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)

def _inputs(eval_data, batch_size, numPrThread=16):
  """Construct input for umd evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # filenames = ["/raid5/hasnat/tf/vmf/dataset/mnist_test.bin"]
  filenames = ["/home/hasnat/Desktop/mnist_verify/vae_mnist/mnist_test.bin"]
  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
  
  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_mnist(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  # Image processing for evaluation.
  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.scalar_mul(0.0039, reshaped_image)

  # Ensure that the random shuffling has good mixing properties.
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False, numPrThread=numPrThread)
##### call from external functions
def distorted_inputs(randFlip=True):
  images, labels = _distorted_inputs(batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

def inputs(eval_data, numPrThread=16):
  images, labels = _inputs(eval_data=eval_data, batch_size=FLAGS.batch_size, numPrThread=numPrThread)

  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels
