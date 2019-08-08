#!/usr/bin/env python

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

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import json
import time

from flask import Flask, request
import numpy as np

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

app = Flask(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    
    parser.add_argument(
        '--port',
        dest='port',
        help='servicer port',
        default=8888,
        type=int
    )
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default="./pre_pandas/pandas.yaml",
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default="./pre_pandas/model_final.pkl",
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/opt/lampp/htdocs/aiimg',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: pdf)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--red-border',
        dest='red_border',
        help='output image file format (default: pdf)',
        default='NG',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )
    parser.add_argument(
        '--is-save',
        dest='is_save',
        help='is_save_result_image',
        default=1,
        type=int
    )
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()


@app.route('/pandas/', methods=['GET', 'POST'])
def main():
    score = [0.3,0.1,0.3,0.1,0.1,0.1,0.3,0.1,0.1,0.1,0.1]
    file = request.get_data()
    img = np.asarray(bytearray(file), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    timers = defaultdict(Timer)
    t = time.time()
    
    print("name:" + str(int(t)))
    start_time = time.time()
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
            model, img, None, timers=timers)

    logger.info('Inference time: {:.3f}s'.format(time.time() - t))
    print("Time:", time.time() - t)
    for k, v in timers.items():
        logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

    ret = {"num": 0, "label_str":"OK,","points":[],"img_name":str(int(t)) + "." + args.output_ext, "process_time" : "s"}    #######
    count = 0 
    for m, boxTemp in enumerate(cls_boxes[1:]):
        #print(boxTemp)
        m_ = m + 1
        if len(boxTemp) == 0:
            pass
        else:
            points = []
            count_this_ng = 0
            for n, box in enumerate(boxTemp): 
                cls_name = str(dummy_coco_dataset['classes'][m_])
                color = (0, 255, 0)
                points_ = []
                
                if float(box[4]) > float(score[m_]):
                    count = count + 1
                    cv2.putText(img, str(round(box[4],2)) + ":" + cls_name, (int(box[0]), int(box[1])),cv2.FONT_HERSHEY_SIMPLEX, 2, color,2)
                    cv2.rectangle(img, (int(box[0]), int(box[1])),(int(box[2]), int(box[3])), color,2)
                    points_ = [int(box[0]), int(box[1]),int(box[2]), int(box[3]), float('%.2f' % box[4])]
                    points_.append(cls_name)
                    ret["num"] = count
                    ret["points"].append(points_)
                    if count_this_ng == 0:
                        ret["label_str"] = ret["label_str"]  +  cls_name + ","
                        count_this_ng += 1
                    else : pass 

    out_put_path = args.output_dir + "/" + str(int(t)) + "." + args.output_ext
    print(out_put_path)
    cv2.imwrite(out_put_path, img)
    end_time = time.time()
    total_time = end_time - start_time
    ret["process_time"] = total_time
    ret["points"] = str(ret["points"])
    ret = json.dumps(ret)
    return ret

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()
    
    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'
    logger = logging.getLogger(__name__)
    app.run(host="0.0.0.0", port= args.port)
