import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from nets import nets_factory
from data import get_split, load_batch, load_labels_into_dict
import os
import time
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
slim = tf.contrib.slim


items_to_descriptions = {'image': 'A 3-channel RGB coloured image', 'label': 'Img label'}

#================ DATASET INFORMATION ======================
#State dataset directory where the tfrecord files are located
tf.app.flags.DEFINE_string('dataset_dir', 'None', 'dataset directory where the tfrecord files are located')
tf.app.flags.DEFINE_string('log_dir', '/tmp/tflog/', 'direcotry where to create log files')
tf.app.flags.DEFINE_string('checkpoint_path', None, 'location of checkpoint file')
tf.app.flags.DEFINE_integer('image_size', 224, 'image size to resize input images to')
tf.app.flags.DEFINE_integer('num_classes', 200, 'number of classes to predict')
tf.app.flags.DEFINE_string('labels_file', None, 'path to labels file')
tf.app.flags.DEFINE_string('file_pattern', 'movidius_%s_*.tfrecord', 'file pattern of TFRecord files')
tf.app.flags.DEFINE_string('file_pattern_for_counting', 'movidius', 'identify tfrecord files')
tf.app.flags.DEFINE_string('preprocessing', 'inception', 'define preprocessing function to use defiend in preprocessing_factory')
#================ TRAINING INFORMATION ======================

tf.app.flags.DEFINE_integer('num_epochs', 50, 'number of epochs to train')
tf.app.flags.DEFINE_integer('batch_size', 50, 'batch size')
tf.app.flags.DEFINE_float('learning_rate', .0002, 'initial learning rate')
tf.app.flags.DEFINE_float('learning_rate_decay', 0.7, 'learning_rate decay factor')
tf.app.flags.DEFINE_integer('epochs_before_decay', 2, 'number of epochs before decay')
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None, 'checkpoint scopes to exclude')
tf.app.flags.DEFINE_string('model_name', 'mobilenet_v1', 'The name of the architecture to train.')
tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_integer('log_every_n_steps', 100, 'The frequency with which logs are print.')
tf.app.flags.DEFINE_boolean('ignore_missing_vars', False, 'When restoring a checkpoint would ignore missing variables.')
tf.app.flags.DEFINE_float('class_weight', None, 'Class weight of the positive weight (binary only)')
tf.app.flags.DEFINE_boolean('save_every_epoch', False, 'Whether to save ckpt file every epoch')
tf.app.flags.DEFINE_string('trainable_scopes', None, 'Comma-separated list of scopes to filter the set of variables to train.' 'By default, None would train all the variables.')
tf.app.flags.DEFINE_string('tb_logdir', None, 'TensorBoard logdir')
FLAGS = tf.app.flags.FLAGS

#LOGDIR = "/home/ubuntu/TF-Movidius-Finetune-TB/TB_logdir"

def _get_variables_to_train():
    """Returns a list of variables to train.

    Returns:
        A list of variables to train by the optimizer.
    """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    if len(variables_to_train) == 0:
    	return None
    return variables_to_train

#Defines functions to load checkpoint
def _get_init_fn():
    """Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
        An init function run by the supervisor.
    """
    if FLAGS.checkpoint_path is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.log_dir):
        tf.logging.info('Ignoring --checkpoint_path because a checkpoint already exists in {}'.format(FLAGS.log_dir))
        tf.logging.warning('Warning --checkpoint_exclude_scopes used when restoring from fine-tuned model {}'.format(FLAGS.checkpoint_exclude_scopes))
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.log_dir)
    elif tf.train.checkpoint_exists(FLAGS.checkpoint_path):
        checkpoint_path = FLAGS.checkpoint_path
    else:
        raise ValueError('No valid checkpoint found in --log_dir or --checkpoint_path: {}, {}'.format(FLAGS.log_dir, FLAGS.checkpoint_path))
    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    tf.logging.info('Fine-tuning from {}'.format(checkpoint_path))

    return slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=FLAGS.ignore_missing_vars)


#Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
#def train_step(sess, train_op, global_step, metrics_op, log=False):
def _train_step(sess, train_op, global_step, log=False):
    '''
    Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
    '''
    #Check the time for each sess run
    start_time = time.time()
    total_loss, global_step_count = sess.run([train_op, global_step])
    time_elapsed = time.time() - start_time
    if log:
        logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)
    return total_loss, global_step_count

def _add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.

    Args:
        result_tensor: The new final node that produces results.
        ground_truth_tensor: The node we feed ground truth data
        into.

    Returns:
        Tuple of (evaluation step, prediction).
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(
                    prediction, tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


def main(_):
    #Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)

    labels_dict = load_labels_into_dict(FLAGS.labels_file)
    #======================= TRAINING PROCESS =========================
    #Now we start to construct the graph and build our model
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level

        #####################
        #    Data Loading   #
        dataset = get_split('train', FLAGS.dataset_dir, FLAGS.num_classes, FLAGS.labels_file, file_pattern=FLAGS.file_pattern, file_pattern_for_counting=FLAGS.file_pattern_for_counting)
        images, _, labels = load_batch(dataset, FLAGS.preprocessing, FLAGS.batch_size, FLAGS.image_size)
        num_batches_per_epoch = int(dataset.num_samples / FLAGS.batch_size)
        num_steps_per_epoch = num_batches_per_epoch
        decay_steps = int(FLAGS.epochs_before_decay * num_steps_per_epoch)

        ######################
        # Select the network #
        ######################
        network_fn = nets_factory.get_network_fn(FLAGS.model_name, num_classes=(dataset.num_classes), weight_decay=FLAGS.weight_decay, is_training=True)
        logits, end_points = network_fn(images)
        final_tensor = tf.identity(end_points['Predictions'], name="final_result")
        tf.summary.histogram('activations', final_tensor)
        #Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!) ?
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
        with tf.name_scope('cross_entropy_loss'):
            #Performs the equivalent to tf.nn.sparse__entropy_with_logits but enhanced with checks
            loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits = logits)
            #Optionally calculate a weighted loss
            if FLAGS.class_weight:
                loss = tf.losses.compute_weighted_loss(loss, weights=FLAGS.class_weight)
            total_loss = tf.losses.get_total_loss() #obtain the regularization losses as well

        #Create the global step for monitoring the learning_rate and training.
        global_step = tf.train.get_or_create_global_step()
        #Define decaying learning rate
        lr = tf.train.exponential_decay(learning_rate = FLAGS.learning_rate,
                                        global_step = global_step,
                                        decay_steps = decay_steps,
                                        decay_rate = FLAGS.learning_rate_decay,
                                        staircase = True)
        #TODO: add options to decide optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)
        variables_to_train = _get_variables_to_train()
        train_op = slim.learning.create_train_op(total_loss, optimizer, variables_to_train=variables_to_train)
        accuracy, prediction = _add_evaluation_step(final_tensor, one_hot_labels)
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()

	#Save summaries and graph for TensorBoard visualization
	#sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter(FLAGS.tb_logdir)
	#writer.add_graph(sess.graph)

        sv = tf.train.Supervisor(logdir = FLAGS.log_dir, summary_op = None, init_fn=_get_init_fn())
        #Run the managed session
        with sv.managed_session() as sess:
	    writer.add_graph(sess.graph)
            for step in range(num_steps_per_epoch * FLAGS.num_epochs):
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, FLAGS.num_epochs)
                    learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                    logging.info('Current Learning Rate: %s', learning_rate_value)
                    logging.info('Current Batch Accuracy: %s', accuracy_value)
                    if FLAGS.save_every_epoch:
                        sv.saver.save(sess, sv.save_path, global_step = sv.global_step)
                if step % FLAGS.log_every_n_steps == 0:
                    loss, _ = _train_step(sess, train_op, sv.global_step, log=True)
		    learning_rate_value, accuracy_value, summaries = sess.run([lr, accuracy, my_summary_op])
		    writer.add_summary(summaries, step)
                    #summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)
                else:
                    loss, _ = _train_step(sess, train_op, sv.global_step)
            #Log the final training loss and train accuracy
            logging.info('Final Loss: %s', loss)
            logging.info('Final Train Accuracy: %s', sess.run(accuracy))
            logging.info('Finished training! Saving model to disk now {}'.format(sv.save_path))
            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)

if __name__ == '__main__':
  tf.app.run()
