# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Utilities driving the train_net binary"""




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from shutil import copyfile
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import numpy as np
import os
import re



from caffe2.python import memonger
from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.core.config import get_output_dir
from detectron.datasets.roidb import combined_roidb_for_training
from detectron.modeling import model_builder
from detectron.utils import lr_policy
from detectron.utils.training_stats import TrainingStats
import detectron.utils.env as envu
import detectron.utils.net as nu
if 1:
    from detectron.datasets import json_dataset
    from detectron.datasets import roidb as roidb_utils
    import detectron.modeling.FPN as fpn
    import detectron.roi_data.fast_rcnn as fast_rcnn_roi_data
    import detectron.utils.blob as blob_utils
Train_1 = 0
if 1:
    def collect(inputs, is_training):
        cfg_key = 'TRAIN' if is_training else 'TEST'
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        k_max = cfg.FPN.RPN_MAX_LEVEL
        k_min = cfg.FPN.RPN_MIN_LEVEL
        num_lvls = k_max - k_min + 1
        roi_inputs = inputs[:num_lvls]
        score_inputs = inputs[num_lvls:]
        if is_training:
            score_inputs = score_inputs[:-2]

        # rois are in [[batch_idx, x0, y0, x1, y2], ...] format
        # Combine predictions across all levels and retain the top scoring
        rois = np.concatenate([blob for blob in roi_inputs])
        scores = np.concatenate([blob for blob in score_inputs]).squeeze()
        inds = np.argsort(-scores)[:post_nms_topN]
        rois = rois[inds, :]
        return rois


    def distribute(rois, label_blobs, outputs, train):
        """To understand the output blob order see return value of
        detectron.roi_data.fast_rcnn.get_fast_rcnn_blob_names(is_training=False)
        """
        lvl_min = cfg.FPN.ROI_MIN_LEVEL
        lvl_max = cfg.FPN.ROI_MAX_LEVEL
        lvls = fpn.map_rois_to_fpn_levels(rois[:, 1:5], lvl_min, lvl_max)

        outputs[0].reshape(rois.shape)
        outputs[0].data[...] = rois

        # Create new roi blobs for each FPN level
        # (See: modeling.FPN.add_multilevel_roi_blobs which is similar but annoying
        # to generalize to support this particular case.)
        rois_idx_order = np.empty((0,))
        for output_idx, lvl in enumerate(range(lvl_min, lvl_max + 1)):
            idx_lvl = np.where(lvls == lvl)[0]
            blob_roi_level = rois[idx_lvl, :]
            outputs[output_idx + 1].reshape(blob_roi_level.shape)
            outputs[output_idx + 1].data[...] = blob_roi_level
            rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
        rois_idx_restore = np.argsort(rois_idx_order)
        blob_utils.py_op_copy_blob(rois_idx_restore.astype(np.int32), outputs[-1])
def train_model():
    """Model training loop."""
    logger = logging.getLogger(__name__)
    model, weights_file, start_iter, checkpoints, output_dir = create_model()   #for create model
    if 'final' in checkpoints:
        # The final model was found in the output directory, so nothing to do
        return checkpoints
    if 0:
        output_dir = '/home/icubic/daily_work/code/Detectron/train/coco_2014_train_ET_PH_part/generalized_rcnn_multi/'
    #output_dir = output_dir + '_101'
    setup_model_for_training(model, weights_file, output_dir)
    training_stats = TrainingStats(model)
    uuuu = model.roi_data_loader._blobs_queue_name
    CHECKPOINT_PERIOD = int(cfg.TRAIN.SNAPSHOT_ITERS / cfg.NUM_GPUS)
    print('------------train.py')
    for cur_iter in range(start_iter, cfg.SOLVER.MAX_ITER):
        training_stats.IterTic()
        lr = model.UpdateWorkspaceLr(cur_iter, lr_policy.get_lr_at_iter(cur_iter))
        #aaa_debug = workspace.FetchBlob('gpu_0/data')
        #bbb_debug = workspace.FetchBlob('gpu_0/conv1_w')
        #ccc_debug = workspace.FetchBlob('gpu_0/'+uuuu)
        try:
            workspace.RunNet(model.net.Proto().name)

            if 0:
                #import detectron.utils.blob as blob_utils
                inputs = [workspace.FetchBlob("gpu_0/rpn_rois_fpn2"),workspace.FetchBlob("gpu_0/rpn_rois_fpn3"),workspace.FetchBlob("gpu_0/rpn_rois_fpn4"),workspace.FetchBlob("gpu_0/rpn_rois_fpn5"), \
                          workspace.FetchBlob("gpu_0/rpn_rois_fpn6"),workspace.FetchBlob("gpu_0/rpn_roi_probs_fpn2"),workspace.FetchBlob("gpu_0/rpn_roi_probs_fpn3"),workspace.FetchBlob("gpu_0/rpn_roi_probs_fpn4"), \
                          workspace.FetchBlob("gpu_0/rpn_roi_probs_fpn5"),workspace.FetchBlob("gpu_0/rpn_roi_probs_fpn6"),workspace.FetchBlob("gpu_0/roidb"),workspace.FetchBlob("gpu_0/im_info"),\
                          ]
                rois = collect(inputs, True)
                #inputs.append(workspace.FetchBlob("gpu_0/rpn_rois_fpn2"))
                im_info = inputs[-1]
                im_scales = im_info[:, 2]
                roidb = blob_utils.deserialize(inputs[-2])
                # For historical consistency with the original Faster R-CNN
                # implementation we are *not* filtering crowd proposals.
                # This choice should be investigated in the future (it likely does
                # not matter).
                json_dataset.add_proposals(roidb, rois, im_scales, crowd_thresh=0)
                roidb_utils.add_bbox_regression_targets(roidb)
                # Compute training labels for the RPN proposals; also handles
                # distributing the proposals over FPN levels
                output_blob_names = fast_rcnn_roi_data.get_fast_rcnn_blob_names()
                blobs = {k: [] for k in output_blob_names}
                fast_rcnn_roi_data.add_fast_rcnn_blobs(blobs, im_scales, roidb)
                for i, k in enumerate(output_blob_names):
                    blob_utils.py_op_copy_blob(blobs[k], outputs[i])
            #if (np.sum(bb == 1))>0:
             #   print('cc')
        except:
            aa = workspace.FetchBlob("gpu_0/rpn_rois_fpn2")
            aaa_debug = workspace.FetchBlob('gpu_0/data')
            print('aaaaaerror')
        #print("blobs:\n{}".format(workspace.Blobs()))
        #print('train.py   aaaaaaaa_debug')
        if 1:


            aaa = workspace.FetchBlob("gpu_0/data")  # nchw
            #img = aaa[1].copy()
            # BGR HWC -> CHW  12
            #transform_img = img.swapaxes(0, 1).swapaxes(1, 2)


            #cv2.imshow("image0 ", transform_img[:, :, (2, 1, 0)])

            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #cv2.imshow('/home/icubic/daily_work/code/Detectron/aaa.png', aaa[0])
            aaa_debug = workspace.FetchBlob('gpu_0/data')
            bbb_debug = workspace.FetchBlob('gpu_0/conv1_w')
            ccc_debug = workspace.FetchBlob('gpu_0/' + uuuu)
            ddd_debug = workspace.FetchBlob('gpu_0/roidb')
            eee_debug = workspace.FetchBlob('gpu_0/im_info')
            #print("Fetched data:\n{}".format(workspace.FetchBlob("gpu_0/data")))
        if cur_iter == start_iter:
            nu.print_net(model)
        training_stats.IterToc()
        training_stats.UpdateIterStats()
        training_stats.LogIterStats(cur_iter, lr)

        if (cur_iter + 1) % (CHECKPOINT_PERIOD/4) == 0 and cur_iter > start_iter:#((cur_iter + 1) % (CHECKPOINT_PERIOD/1) == 0 and (cur_iter > start_iter and cur_iter < 50000)) or ((cur_iter + 1) % (CHECKPOINT_PERIOD/8) == 0 and cur_iter > 50000):
            checkpoints[cur_iter] = os.path.join(
                output_dir, 'model_iter_50_{}.pkl'.format(cur_iter)
            )
            nu.save_model_to_weights_file(checkpoints[cur_iter], model)

        if cur_iter == start_iter + training_stats.LOG_PERIOD:
            # Reset the iteration timer to remove outliers from the first few
            # SGD iterations
            training_stats.ResetIterTimer()

        if np.isnan(training_stats.iter_total_loss):
            logger.critical('Loss is NaN, exiting...')
            model.roi_data_loader.shutdown()
            envu.exit_on_error()

    # Save the final model
    checkpoints['final'] = os.path.join(output_dir, 'model_final_50.pkl')
    nu.save_model_to_weights_file(checkpoints['final'], model)
    # Shutdown data loading threads
    model.roi_data_loader.shutdown()
    return checkpoints


def create_model():
    """Build the model and look for saved model checkpoints in case we can
    resume from one.
    """
    logger = logging.getLogger(__name__)
    start_iter = 0
    checkpoints = {}
    output_dir = get_output_dir(cfg.TRAIN.DATASETS, training=True)
    weights_file = cfg.TRAIN.WEIGHTS#'/home/icubic/daily_work/code/Detectron/train/coco_2014_train/generalized_rcnn_S1S2/model_final2.pkl'##cfg.TRAIN.WEIGHTS
    if cfg.TRAIN.AUTO_RESUME:
        # Check for the final model (indicates training already finished)
        final_path = os.path.join(output_dir, 'model_final.pkl')
        if os.path.exists(final_path):
            logger.info('model_final.pkl exists; no need to train!')
            return None, None, None, {'final': final_path}, output_dir

        if cfg.TRAIN.COPY_WEIGHTS:
            copyfile(
                weights_file,
                os.path.join(output_dir, os.path.basename(weights_file)))
            logger.info('Copy {} to {}'.format(weights_file, output_dir))

        # Find the most recent checkpoint (highest iteration number)
        files = os.listdir(output_dir)
        for f in files:
            iter_string = re.findall(r'(?<=model_iter)\d+(?=\.pkl)', f)
            if len(iter_string) > 0:
                checkpoint_iter = int(iter_string[0])
                if checkpoint_iter > start_iter:
                    # Start one iteration immediately after the checkpoint iter
                    start_iter = checkpoint_iter + 1
                    resume_weights_file = f

        if start_iter > 0:
            # Override the initialization weights with the found checkpoint
            weights_file = os.path.join(output_dir, resume_weights_file)
            logger.info(
                '========> Resuming from checkpoint {} at start iter {}'.
                format(weights_file, start_iter)
            )

    logger.info('Building model: {}'.format(cfg.MODEL.TYPE))
    model = model_builder.create(cfg.MODEL.TYPE, train=True)   #for create model
    if cfg.MEMONGER:
        optimize_memory(model)
    # Performs random weight initialization as defined by the model
    workspace.RunNetOnce(model.param_init_net)
    return model, weights_file, start_iter, checkpoints, output_dir


def optimize_memory(model):
    """Save GPU memory through blob sharing."""
    for device in range(cfg.NUM_GPUS):
        namescope = 'gpu_{}/'.format(device)
        losses = [namescope + l for l in model.losses]
        model.net._net = memonger.share_grad_blobs(
            model.net,
            losses,
            set(model.param_to_grad.values()),
            namescope,
            share_activations=cfg.MEMONGER_SHARE_ACTIVATIONS
        )


def setup_model_for_training(model, weights_file, output_dir):
    """Loaded saved weights and create the network in the C2 workspace."""
    logger = logging.getLogger(__name__)
    queue_name = add_model_training_inputs(model)    #non with pre_data

    if weights_file:
        # Override random weight initialization with weights from a saved model
        nu.initialize_gpu_from_weights_file(model, weights_file, gpu_id=0)
    # Even if we're randomly initializing we still need to synchronize
    # parameters across GPUs

    nu.broadcast_parameters(model)
    workspace.CreateNet(model.net)
    #return





    logger.info('Outputs saved to: {:s}'.format(os.path.abspath(output_dir)))
    dump_proto_files(model, output_dir)

    # Start loading mini-batches and enqueuing blobs
    model.roi_data_loader.register_sigint_handler()
    model.roi_data_loader.start(prefill=True)

    return output_dir


def add_model_training_inputs(model):#for add input data
    """Load the training dataset and attach the training inputs to the model."""
    logger = logging.getLogger(__name__)
    logger.info('Loading dataset: {}'.format(cfg.TRAIN.DATASETS))
    roidb = combined_roidb_for_training(
        cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES
    )
    logger.info('{:d} roidb entries'.format(len(roidb)))
    queue_name = model_builder.add_training_inputs(model, roidb=roidb)   #for roidb input_important
    return queue_name


def dump_proto_files(model, output_dir):
    """Save prototxt descriptions of the training network and parameter
    initialization network."""
    with open(os.path.join(output_dir, 'net.pbtxt'), 'w') as fid:
        fid.write(str(model.net.Proto()))
    with open(os.path.join(output_dir, 'param_init_net.pbtxt'), 'w') as fid:
        fid.write(str(model.param_init_net.Proto()))
