import os
import sys
import tensorflow as tf
import click
import mlflow
from tf_estimator.session_log_hook import SessionLogHook, SessionLogEndHook

worker_index_id = 0
num_gpus = 1


def get_input_fn_truple(num_workers=1, worker_index=0):
  image_list = os.path.join(cfg.dataset.LIST_DIR, cfg.dataset.train_list)
  
  dataset = tf.data.TextLineDataset(image_list)
  dataset = dataset.shard(num_workers, worker_index)

  dataset = dataset.shuffle(10000)
  dataset = dataset.repeat()

  dataset = dataset.map(lambda image_id:
                        tuple(tf.py_func(get_blobs_func, [image_id],
                                         [tf.float32, tf.float32,
                                          tf.float32, tf.string])),
                        num_parallel_calls=2)
  dataset = dataset.filter(lambda x, y, z, s:
                           tf.not_equal(tf.shape(x)[0], 0))

  dataset = dataset.prefetch(20)

  iterator = dataset.make_one_shot_iterator()

  image, im_info, label, image_id = iterator.get_next()
  image.set_shape([1, None, None, 3])
  im_info.set_shape([3])
  label.set_shape([None, 5])
  return image, im_info, label, image_id, worker_index


def model_fn(features, labels, mode):
  global worker_index_id
  ''' Contruct graph '''
  tf.set_random_seed(cfg.RNG_SEED)

  image, im_info, label, image_id, worker_index = get_input_fn_truple(num_workers=num_gpus, worker_index=worker_index_id)
  
  worker_index_id += 1

  from nets.resnet import resnetv
  
  net = resnetv(num_layers=cfg.RESNET.num_layers)
  layers = net.create_architecture('TRAIN', cfg.dataset.NUM_CLASSES,
                                   tag='default',
                                   anchor_scales=cfg.ANCHOR_SCALES,
                                   anchor_ratios=cfg.ANCHOR_RATIOS,
                                   image=image,
                                   im_info=im_info,
                                   gt_boxes=label)

  loss = layers['total_loss']
  global_step = tf.train.get_or_create_global_step()

  base_lr = cfg.TRAIN.LEARNING_RATE * num_gpus
  warmup_iters = 1000
  lr = tf.train.polynomial_decay(
    base_lr, global_step,
    decay_steps=cfg.TRAIN.MAX_ITERS,
    end_learning_rate=0, power=0.9,
    name='learning_rate')

  lr = tf.minimum(lr, base_lr / warmup_iters * tf.to_float(global_step))

  optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)
  optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

  gvs = optimizer.compute_gradients(loss)
  train_op = optimizer.apply_gradients(gvs, global_step)

  
  exclude = ['global_step', 'learning_rate']
  exclude += ['tower_{}/learning_rate'.format(i) for i in range(num_gpus)]
  variables_to_restore = tf.contrib.slim.get_variables_to_restore(
    exclude=exclude)
  tf.train.init_from_checkpoint(cfg.TRAIN.finetune_model,
                                {v.name.split(':')[0]: v for v in variables_to_restore}
                                )

  tf.summary.scalar('learning_rate', lr)

  return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)


@click.command()
def train(data_folder, list_folder, enable_summary, config, gpus):
  update_config(config, data_folder, list_folder)
  os.environ['CUDA_VISIBLE_DEVICES'] = gpus
  global num_gpus
  num_gpus = len(gpus.split(','))

  image_list = os.path.join(cfg.dataset.LIST_DIR,
                            cfg.dataset.train_list)
  with open(image_list) as f:
    image_num = len(f.readlines())
  num_per_epoch = image_num / num_gpus

  cfg.TRAIN.MAX_ITERS = num_per_epoch * 15

  mlflow.log_param('train', cfg.TRAIN)
  pprint.pprint(cfg)

  run_config = tf.estimator.RunConfig(
    save_checkpoints_steps=cfg.TRAIN.snapshot_interval,
    log_step_count_steps=(cfg.TRAIN.log_interval if enable_summary else cfg.TRAIN.MAX_ITERS),
    keep_checkpoint_max=cfg.TRAIN.snapshot_max_to_keep,
    session_config=tf.ConfigProto(
      allow_soft_placement=True,
      gpu_options=tf.GPUOptions(force_gpu_compatible=True, allow_growth=True))
  )

  mod_fn = model_fn
  if num_gpus > 1:
    mod_fn = tf.contrib.estimator.replicate_model_fn(
      mod_fn,
      loss_reduction=tf.losses.Reduction.MEAN)

  estimator = tf.estimator.Estimator(
    model_fn=mod_fn,
    model_dir=cfg.TRAIN.snapshots_dir,
    config=run_config,
  )

  session_log_hook = SessionLogHook(cfg.TRAIN.log_interval)
  estimator.train(input_fn=lambda: ([0] * num_gpus, [0] * num_gpus),
                  hooks=[session_log_hook],
                  max_steps=cfg.TRAIN.MAX_ITERS)

  mlflow.log_artifact(cfg.TRAIN.snapshots_dir + '/checkpoint', "train_model")


if __name__ == '__main__':
  train()
