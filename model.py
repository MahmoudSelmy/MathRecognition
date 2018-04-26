import tensorflow as tf
import random
import cv2
import numpy as np
import sys
import math
import os


def _find_image_files(data_dir, labels_file, split_ratio=(0.90, 0.05, 0.05)):
    """Build a list of all images files and labels in the data set.

    Args:
    data_dir: string, path to the root directory of images.

      Assumes that the image data set resides in img_extension files located in
      the following directory structure.

        data_dir/dog/another-image.img_extension
        data_dir/dog/my-image.jpg

      where 'dog' is the label associated with these images.

    labels_file: string, path to the labels file.

      The list of valid labels are held in this file. Assumes that the file
      contains entries as such:
        dog
        cat
        flower
      where each line corresponds to a label. We map each label contained in
      the file to an integer starting with the integer 0 corresponding to the
      label contained in the first line.

    Returns:
    addrs: list of strings; each string is a path to an image file.
    texts: list of strings; each string is the class, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth.
    """
    print('Determining list of input files and labels from %s.' % data_dir)
    unique_labels = [l.strip() for l in tf.gfile.FastGFile(labels_file, 'r').readlines()]

    labels = []
    addrs = []
    # texts = []

    # Leave label index 0 empty as a background class.
    label_index = 0

    # Construct the list of img_extension files and labels.
    for text in unique_labels:
        print(text, label_index)
        img_extension_file_path = '%s/%s/*' % (data_dir, text)
        matching_files = tf.gfile.Glob(img_extension_file_path)

        labels.extend([label_index] * len(matching_files))
        # texts.extend([text] * len(matching_files))
        addrs.extend(matching_files)

        label_index += 1

    print("addrs list length:", len(addrs))

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(addrs)))
    # random.seed(12345)
    random.shuffle(shuffled_index)  # shuffles in place and returns None

    addrs = [addrs[i] for i in shuffled_index]
    # texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    ###################################################
    # Divide the hata into 90% train, 5% validation, and 5% test
    train_addrs = addrs[0:int(split_ratio[0] * len(addrs))]
    train_labels = labels[0:int(split_ratio[0] * len(labels))]

    val_addrs = addrs[int(split_ratio[0] * len(addrs)):int((split_ratio[0] + split_ratio[1]) * len(addrs))]
    val_labels = labels[int(split_ratio[0] * len(addrs)):int((split_ratio[0] + split_ratio[1]) * len(addrs))]

    test_addrs = addrs[int((split_ratio[0] + split_ratio[1]) * len(addrs)):]
    test_labels = labels[int((split_ratio[0] + split_ratio[1]) * len(labels)):]
    ####################################################

    for i, j in zip(addrs, labels):
        print(i, "--->", j)

    print('Found %d img_extension files across %d labels inside %s.' % (len(addrs), len(unique_labels), data_dir))
    return [train_addrs, train_labels, val_addrs, val_labels, test_addrs, test_labels]


def load_image(addr, height=False, width=False, rgb=True):
    # read an image and resize to (height, width)
    if rgb:
        # cv2 load images as BGR, convert it to RGB
        img = cv2.imread(addr)
        if height and width:
            img = cv2.resize(img, (height, width), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
    else:
        # Grayscale
        img = cv2.imread(addr, cv2.IMREAD_GRAYSCALE)
        if height and width:
            img = cv2.resize(img, (height, width), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32)

    return img


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def classification_tfrecord(tfrecord_path, addrs_list, labels_list, height, width, rgb=True):
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    for i in range(len(addrs_list)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print(tfrecord_path + 'data: {}/{}'.format(i, len(addrs_list)))
            sys.stdout.flush()

        # Load the image
        img = load_image(addrs_list[i], height, width, rgb)

        label = labels_list[i]

        # Create a feature
        feature = {'record/label': _int64_feature(label),
                   'record/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


def train_valid_test_classification_tfrecords(data_dir, labels_file, height=False, width=False, rgb=True, name="",
                                              split_ratio=(0.90, 0.05, 0.05)):
    train_addrs, train_labels, val_addrs, val_labels, test_addrs, test_labels = _find_image_files(data_dir,
                                                                                                  labels_file,
                                                                                                  split_ratio=split_ratio)

    classification_tfrecord(name + "training.tfrecords", train_addrs, train_labels, height, width, rgb)
    classification_tfrecord(name + "validation.tfrecords", val_addrs, val_labels, height, width, rgb)
    classification_tfrecord(name + "test.tfrecords", test_addrs, test_labels, height, width, rgb)


def batch_from_tfrecord(data_path, height, width, sess, rgb=True, num_epochs=1, batch_size=1, capacity=100,
                        num_threads=1, min_after_dequeue=50):
    feature = {'record/image': tf.FixedLenFeature([], tf.string),
               'record/label': tf.FixedLenFeature([], tf.int64)}

    # Create a list of filenames and pass it to a queue
    if num_epochs is None:
        filename_queue = tf.train.string_input_producer([data_path])
    else:
        filename_queue = tf.train.string_input_producer([data_path], num_epochs)

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['record/image'], tf.float32)
    #print(image)

    # Cast label data into int32
    label = tf.cast(features['record/label'], tf.int32)
    #print(label)

    # Reshape image data into the original shape
    if rgb:
        image = tf.reshape(image, [height, width, 3])
    else:
        image = tf.reshape(image, [height, width, 1])

    # Any preprocessing here ...

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                            num_threads=num_threads, min_after_dequeue=min_after_dequeue)

    return  images, labels



def weight_variable(shape, stddev, name):
    initial = tf.truncated_normal(shape, stddev=stddev)
    var = tf.Variable(initial, name=name)
    return var


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def inference(x):
    W_conv1 = weight_variable([5, 5, 1, 32], 0.1, "W_conv1")
    b_conv1 = bias_variable([32], "b_conv1")
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64], 0.1, "W_conv2")
    b_conv2 = bias_variable([64], "b_conv2")
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    W_fc1 = weight_variable([12 * 12 * 64, 1024], 0.1, "W_fc1")
    b_fc1 = bias_variable([1024], "b_fc1")
    h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    W_fc2 = weight_variable([1024,25], 0.1, "W_fc2")
    b_fc2 = bias_variable([25], "b_fc2")
    softmax_linear = tf.add(tf.matmul(h_fc1, W_fc2), b_fc2)

    return softmax_linear


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'))




def training(loss):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return train_step


def evaluation(logits, y_):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

    return accuracy





def train(num_epochs, batch_size, model_path=None):
    sess = tf.Session()
    no_training_examples = sum(1 for _ in tf.python_io.tf_record_iterator("training.tfrecords"))
    print("training set has:", no_training_examples, "examples")

    images, labels = batch_from_tfrecord("training.tfrecords", 45, 45, sess, rgb=False, num_epochs=None,
                                                         batch_size=batch_size, capacity=1000, num_threads=1, min_after_dequeue=50)

    images_val, labels_val = batch_from_tfrecord("validation.tfrecords", 45, 45, sess, rgb=False,
                                                         num_epochs=None,
                                                         batch_size=batch_size*10, capacity=10000, num_threads=1,
                                                         min_after_dequeue=50)

    ######################################################
    x = tf.placeholder(tf.float32, shape=[None, 45, 45, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 25])




    logits = inference(x)
    losses = loss(logits, y_)
    train_step = training(losses)
    accuracy = evaluation(logits, y_)

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.Saver()

    if model_path is None:
        sess.run(tf.initialize_all_variables())
    else:
        saver.restore(sess, model_path)


    batch_index = 0
    prev_test_accuracy = 0.0
    print("/////////////////////////////////////////////////////////")

    try:
        for epoch in range(1,num_epochs+1):
            for batches in range(int(no_training_examples/batch_size)):
                batch_index += 1

                img, lbl = sess.run([images, labels])
                img = img.astype(np.uint8)
                lbl = np.eye(25)[[lbl]]

                _, training_accuracy = sess.run([train_step,accuracy], feed_dict={x: img, y_: lbl})

                print("##### batch number:" + str(batch_index), "training_accuracy:" + str(training_accuracy))

                if batch_index % 25 == 0:
                    img, lbl = sess.run([images_val, labels_val])
                    img = img.astype(np.uint8)
                    lbl = np.eye(25)[[lbl]]

                    test_accuracy = sess.run(accuracy, feed_dict={x: img, y_: lbl})
                    print("##### batch number:" + str(batch_index), "validation_accuracy:" + str(test_accuracy))


                    if test_accuracy > prev_test_accuracy:
                        #save here
                        save_path = saver.save(sess, save_path="/home/barakat/Desktop/tf_self_framework/utils/ckpts/convnet", write_meta_graph=False)
                        print("Model saved in file: %s" % save_path)

                    prev_test_accuracy = test_accuracy

            print("////////////////////////////////////////////")
            print("End of epoch", epoch, "out of", num_epochs)
        print("All", num_epochs, "are done")


    finally:
        coord.request_stop()
        coord.join(threads)
        print("coordinator killed all threads successfully")





    ###############################################################



if __name__ == "__main__":
    train(1, 512, model_path="/home/barakat/Desktop/tf_self_framework/utils/ckpts/convnet")
    #train(1, 512)