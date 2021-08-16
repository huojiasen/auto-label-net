import sys
import shutil
import click

import mlflow
import tensorflow as tf
from tensorflow.saved_model import tag_constants
from tensorflow.saved_model import signature_constants


from model.nms_wrapper import nms
from model.nms_wrapper import soft_nms

import numpy as np
import os, cv2
from tqdm import tqdm

from time import gmtime, strftime
import json

MULTIHEAD_NAME_LIST = ["shape_prob", "color_prob", "bbox_pred", "rois"]
SINGLEHEAD_NAME_LIST = ["cls_prob", "bbox_pred", "rois"]

def matching_three(a_entries, b_entries, c_entries):

  res_entries = []
  for a_ent in a_entries:
    for b_ent in b_entries:
      for c_ent in c_entries:
        if a_ent[3:] == b_ent[3:] and a_ent[3:] == c_ent[3:]:
          res_entries.append([a_ent[0], a_ent[1], b_ent[1], c_ent[1], a_ent[2],
                              b_ent[2], c_ent[2], a_ent[3], a_ent[4], a_ent[5], a_ent[6]])
  return res_entries

def write_json(result_dir, entries):
  objects = []
  json_format = dict()
  json_format['version'] = 0.0
  json_format['tags'] = []
  image_name = ""
  for ent in entries:
    for i in range(4, 11):
      ent[i] = float(ent[i])
  json_format['objects'] = objects

  dst_path = os.path.join(result_dir, image_name[:-3] + "json")
  with open(dst_path, 'w') as f:
    json.dump(json_format, f)

def split_detections_multihead(sess, tensor_dict, image_folder, image_name, result_dir, nms_method='hard'):
    im_path = os.path.join(image_folder, image_name)
    im = cv2.imread(im_path)
    color_scores, shape_scores, boxes = im_detect_multihead_pb(sess, tensor_dict, im)
    final_scores = merge_prob(color_scores, shape_scores)

    color_entries = get_entries(image_name, boxes, color_scores, COLOR_CLASSES, 0.5)
    shape_entries = get_entries(image_name, boxes, shape_scores, SHAPE_CLASSES, 0.5)
    final_entries = get_entries(image_name, boxes, final_scores, CLASSES, 0.5)
    filter_entries = matching_three(color_entries, shape_entries, final_entries)
    write_json(result_dir, filter_entries)

def get_input_output_tensor_dict(model):
  tensor_dict = {}
  inputs = dict(model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs)
  outputs = dict(model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs)

  merge_dict = inputs.copy()
  merge_dict.update(outputs)
  for key, val in merge_dict.items():
    tensor = tf.get_default_graph().get_tensor_by_name(val.name)
    if tensor is None:
      raise TypeError("Expecting tensor " + val.name + " doesn't exist., The key is " + key)
    tensor_dict[key] = tensor
  return  tensor_dict

def run_eval(demonet, tfmodel, im_folder, im_list, result_dir):
  cfg.TEST.HAS_RPN = True  # Use RPN for proposals
  cfg.TEST.TESTING_SABE_PBTXT = True

  # set config
  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True

  # init session
  sess = tf.Session(config=tfconfig)
  model = tf.saved_model.loader.load(export_dir=tfmodel, sess=sess, tags=[tag_constants.SERVING])

  tensor_dict = get_input_output_tensor_dict(model)

  print('Loaded network {:s}'.format(tfmodel))
  count = 0
  with open(im_list, "r") as file:
    content = file.readlines()
    for i in tqdm(content):
      im_name = i.strip() + ".jpg"
      print(im_name)
      print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
      if cfg.TASK.MODEL_TASK ==  "Multi":
        split_detections_multihead(sess, tensor_dict, im_folder, im_name, result_dir)
  sess.close()
  mlflow.log_artifacts(result_dir, "infer_result")

@click.command()
def infer(data_folder, list_folder, infer_folder, model, config):
  update_config(config, data_folder, list_folder)
  image_folder = os.path.join(cfg.dataset.DATA_DIR, cfg.dataset.IMAGE_DIR)
  eval_list = os.path.join(cfg.dataset.LIST_DIR, cfg.dataset.test_list)
  demonet = 'resnet'
  if os.path.exists(infer_folder):
    shutil.rmtree(infer_folder)
  os.makedirs(infer_folder)

  if os.path.exists(infer_folder + "/split"):
      shutil.rmtree(infer_folder + "/split")
  os.makedirs(infer_folder + "/split")

  visual_dir = infer_folder + "/visual"
  if os.path.exists(visual_dir):
    shutil.rmtree(visual_dir)
  os.mkdir(visual_dir)

  run_eval(demonet, model, image_folder, eval_list, infer_folder)

if __name__ == '__main__':
  infer()
