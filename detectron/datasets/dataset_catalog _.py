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

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os


# Path to data dir
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Required dataset entry keys
_IM_DIR = 'image_directory'
_ANN_FN = 'annotation_file'

# Optional dataset entry keys
_IM_PREFIX = 'image_prefix'
_DEVKIT_DIR = 'devkit_directory'
_RAW_DIR = 'raw_dir'

# Available datasets
_DATASETS = {
    'cityscapes_fine_instanceonly_seg_train': {
        _IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        _ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_train.json',
        _RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_val': {
        _IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        # use filtered validation as there is an issue converting contours
        _ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        _RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_test': {
        _IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        _ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_test.json',
        _RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'coco_2014_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_train2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2014.json'
    },
    'coco_2014_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2014.json'
    },
    'coco_2014_minival': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_minival2014.json'
},
    'coco_2014_train_xiongmao': {
        _IM_DIR:
            _DATA_DIR + '/xiongmao/part',
        _ANN_FN:
            _DATA_DIR + '/xiongmao/part/new.json'
    },
    'coco_2014_train_xiongmao_nox7': {
        _IM_DIR:
            _DATA_DIR + '/xiongmao_nox7',
        _ANN_FN:
            _DATA_DIR + '/xiongmao_nox7/new.json'
    },
    'coco_2014_train_xiongmao_multi': {
        _IM_DIR:
            _DATA_DIR + '/xiongmao_multi',
        _ANN_FN:
            _DATA_DIR + '/xiongmao_multi/new.json'
    },
    'coco_2014_train_xiongmaotest_multi': {
        _IM_DIR:
            _DATA_DIR + '/xiongmaotest_multi',
        _ANN_FN:
            _DATA_DIR + '/xiongmaotest_multi/new.json'
    },
    'coco_2014_train_xiongmaotest': {
        _IM_DIR:
            _DATA_DIR + '/xiongmaotest',
        _ANN_FN:
            _DATA_DIR + '/xiongmaotest/new.json'
    },
     'coco_2014_train_dengpao': {
        _IM_DIR:
            _DATA_DIR + '/dengpao',
        _ANN_FN:
            _DATA_DIR + '/dengpao/new.json'
    },
    'coco_2014_train_s1': {
        _IM_DIR:
            _DATA_DIR + '/S1',
        _ANN_FN:
            _DATA_DIR + '/S1/new.json'
    },
    'coco_2014_train_s2': {
        _IM_DIR:
            _DATA_DIR + '/S2',
        _ANN_FN:
            _DATA_DIR + '/S2/new.json'
    },
    'coco_2014_train_s3': {
        _IM_DIR:
            _DATA_DIR + '/S3',
        _ANN_FN:
            _DATA_DIR + '/S3/new.json'
    },
    'coco_2014_train_s4': {
        _IM_DIR:
            _DATA_DIR + '/S4',
        _ANN_FN:
            _DATA_DIR + '/S4/new.json'
    },
    'coco_2014_train_s5': {
        _IM_DIR:
            _DATA_DIR + '/S5',
        _ANN_FN:
            _DATA_DIR + '/S5/new.json'
    },
    'coco_2014_train_xibao': {
        _IM_DIR:
            _DATA_DIR + '/xibao',
        _ANN_FN:
            _DATA_DIR + '/xibao/new.json'
    },
    'coco_2014_train_s5ac': {
        _IM_DIR:
            _DATA_DIR + '/S5ac',
        _ANN_FN:
            _DATA_DIR + '/S5ac/new.json'
    },
    'coco_2014_train_s6': {
        _IM_DIR:
            _DATA_DIR + '/S6',
        _ANN_FN:
            _DATA_DIR + '/S6/new.json'
    },
    'coco_2014_train_guanzi': {
        _IM_DIR:
            _DATA_DIR + '/guanzi',
        _ANN_FN:
            _DATA_DIR + '/guanzi/new.json'
    },
    'coco_2014_train_kahuan': {
        _IM_DIR:
            _DATA_DIR + '/kahuan',
        _ANN_FN:
            _DATA_DIR + '/kahuan/new.json'
    },
    'coco_2014_train_ltps': {
        _IM_DIR:
            _DATA_DIR + '/shanghai/shanghai_ltps',
        _ANN_FN:
            _DATA_DIR + '/shanghai/shanghai_ltps/new.json'
    },
    'coco_2014_train_ET_PH_part': {
        _IM_DIR:
            _DATA_DIR + '/hehui/ET_PH/part/',
        _ANN_FN:
            _DATA_DIR + '/hehui/ET_PH/part/new.json'
    },
    'coco_2014_train_ltps_all': {
        _IM_DIR:
            _DATA_DIR + '/shanghai/shanghai_ltps_all',
        _ANN_FN:
            _DATA_DIR + '/shanghai/shanghai_ltps_all/new.json'
    },
    'coco_2014_train_ltps_res101': {
        _IM_DIR:
            _DATA_DIR + '/shanghai/shanghai_ltps',
        _ANN_FN:
            _DATA_DIR + '/shanghai/shanghai_ltps/new.json'
    },
    'coco_2014_train_linyu': {
        _IM_DIR:
            _DATA_DIR + '/s_linyu',
        _ANN_FN:
            _DATA_DIR + '/s_linyu/new.json'
    },
    'coco_2014_train_ltps_unlabel': {
        _IM_DIR:
            _DATA_DIR + '/shanghai/shanghai_ltps_unlabel',
        _ANN_FN:
            _DATA_DIR + '/shanghai/shanghai_ltps_unlabel/new.json'
    },
    'coco_2014_train_ltps_unlabel1': {
        _IM_DIR:
            _DATA_DIR + '/shanghai/shanghai_ltps_unlabel1',
        _ANN_FN:
            _DATA_DIR + '/shanghai/shanghai_ltps_unlabel1/new.json'
    },
    'coco_2014_train_tp': {
        _IM_DIR:
            _DATA_DIR + '/shanghai/shanghai_tp',
        _ANN_FN:
            _DATA_DIR + '/shanghai/shanghai_tp/new.json'
    },
    'coco_2014_train_mask': {
        _IM_DIR:
            _DATA_DIR + '/shanghai/shanghai_mask',
        _ANN_FN:
            _DATA_DIR + '/shanghai/shanghai_mask/new.json'
    },
    'coco_2014_train_s7': {
        _IM_DIR:
            _DATA_DIR + '/S7',
        _ANN_FN:
            _DATA_DIR + '/S7/new.json'
    },
    'coco_2014_train_s6data1': {
        _IM_DIR:
            _DATA_DIR + '/S6data1',
        _ANN_FN:
            _DATA_DIR + '/S6data1/new.json'
    },
    'coco_2014_train_dianchi': {
        _IM_DIR:
            _DATA_DIR + '/coco-json-export--dingqi01',
        _ANN_FN:
            _DATA_DIR + '/coco-json-export--dingqi01/new.json'
    },
    'coco_2014_train_s6data2': {
        _IM_DIR:
            _DATA_DIR + '/S6data2',
        _ANN_FN:
            _DATA_DIR + '/S6data2/new.json'
    },
    'coco_2014_train_s6data2addtest': {
        _IM_DIR:
            _DATA_DIR + '/S6data2addtest',
        _ANN_FN:
            _DATA_DIR + '/S6data2addtest/new.json'
    },
    'coco_2014_train_s6data3': {
        _IM_DIR:
            _DATA_DIR + '/S6data3',
        _ANN_FN:
            _DATA_DIR + '/S6data3/new.json'
    },
    'coco_2014_train_aoi': {
        _IM_DIR:
            _DATA_DIR + '/aoi/data_all',
        _ANN_FN:
            _DATA_DIR + '/aoi/data_all/new.json'
    },
    'coco_2014_train_zhoucheng': {
        _IM_DIR:
            _DATA_DIR + '/zhoucheng',
        _ANN_FN:
            _DATA_DIR + '/zhoucheng/new.json'
    },
    'coco_2014_train_s6data4': {
        _IM_DIR:
            _DATA_DIR + '/S6data4',
        _ANN_FN:
            _DATA_DIR + '/S6data4/new.json'
    },
    'coco_2014_train_44': {
        _IM_DIR:
            _DATA_DIR + '/A1_S1',
        _ANN_FN:
            _DATA_DIR + '/A1_S1/new.json'
    },
    'coco_2014_train_5': {
        _IM_DIR:
            _DATA_DIR + '/S5',
        _ANN_FN:
            _DATA_DIR + '/S5/new.json'
    },	
    'coco_2014_train_yunsheng': {
        _IM_DIR:
            _DATA_DIR + '/yunsheng',
        _ANN_FN:
            _DATA_DIR + '/yunsheng/new.json'
    },
    'coco_2014_minival_s1': {
        _IM_DIR:
            _DATA_DIR + '/S1',
        _ANN_FN:
            _DATA_DIR + '/S1/new.json'
    },
    'coco_2014_minival_s2': {
        _IM_DIR:
            _DATA_DIR + '/S2',
        _ANN_FN:
            _DATA_DIR + '/S2/new.json'
    },
    'coco_2014_minival_s3': {
        _IM_DIR:
            _DATA_DIR + '/S3',
        _ANN_FN:
            _DATA_DIR + '/S3/new.json'
    },
    'coco_2014_minival_s4': {
        _IM_DIR:
            _DATA_DIR + '/S4',
        _ANN_FN:
            _DATA_DIR + '/S4/new.json'
    },
    'coco_2014_minival_xiongmao': {
        _IM_DIR:
            _DATA_DIR + '/xiongmao/part',
        _ANN_FN:
            _DATA_DIR + '/xiongmao/part/new.json'
    },
    'coco_2014_minival_xibao': {
        _IM_DIR:
            _DATA_DIR + '/xibao',
        _ANN_FN:
            _DATA_DIR + '/xibao/new.json'
    },
    'coco_2014_minival_dengpao': {
        _IM_DIR:
            _DATA_DIR + '/dengpao',
        _ANN_FN:
            _DATA_DIR + '/dengpao/new.json'
    },
    'coco_2014_minival_aoi': {
        _IM_DIR:
            _DATA_DIR + '/aoi/data_all',
        _ANN_FN:
            _DATA_DIR + '/aoi/data_all/new.json'
    },
    'coco_2014_minival_xiongmao_nox7': {
        _IM_DIR:
            _DATA_DIR + '/xiongmao_nox7',
        _ANN_FN:
            _DATA_DIR + '/xiongmao_nox7/new.json'
    },
    'coco_2014_minival_dianzhi': {
        _IM_DIR:
            _DATA_DIR + '/coco-json-export--dingqi01',
        _ANN_FN:
            _DATA_DIR + '/coco-json-export--dingqi01/new.json'
    },
    'coco_2014_minival_xiongmao_multi': {
        _IM_DIR:
            _DATA_DIR + '/xiongmao_multi',
        _ANN_FN:
            _DATA_DIR + '/xiongmao_multi/new.json'
    },
    'coco_2014_minival_kahuan': {
        _IM_DIR:
            _DATA_DIR + '/kahuan',
        _ANN_FN:
            _DATA_DIR + '/kahuan/new.json'
    },
    'coco_2014_minival_ET_PH_part': {
        _IM_DIR:
            _DATA_DIR + '/hehui/ET_PH/part',
        _ANN_FN:
            _DATA_DIR + '/hehui/ET_PH/part/new.json'
    },
    'coco_2014_minval_guanzi': {
        _IM_DIR:
            _DATA_DIR + '/guanzi',
        _ANN_FN:
            _DATA_DIR + '/guanzi/new.json'
    },
    'coco_2014_minival_zhoucheng': {
        _IM_DIR:
            _DATA_DIR + '/zhoucheng',
        _ANN_FN:
            _DATA_DIR + '/zhoucheng/new.json'
    },
    'coco_2014_minival_yunsheng': {
        _IM_DIR:
            _DATA_DIR + '/yunsheng',
        _ANN_FN:
            _DATA_DIR + '/yunsheng/new.json'
    },
    'coco_2014_minival_s5': {
        _IM_DIR:
            _DATA_DIR + '/S5',
        _ANN_FN:
            _DATA_DIR + '/S5/new.json'
    },
    'coco_2014_minival_s5ac': {
        _IM_DIR:
            _DATA_DIR + '/S5ac',
        _ANN_FN:
            _DATA_DIR + '/S5ac/new.json'
    },
    'coco_2014_minival_s6': {
        _IM_DIR:
            _DATA_DIR + '/S6',
        _ANN_FN:
            _DATA_DIR + '/S6/new.json'
    },
    'coco_2014_minival_s7': {
        _IM_DIR:
            _DATA_DIR + '/S7',
        _ANN_FN:
            _DATA_DIR + '/S7/new.json'
    },
    'coco_2014_minival_linyu': {
        _IM_DIR:
            _DATA_DIR + '/s_linyu',
        _ANN_FN:
            _DATA_DIR + '/s_linyu/new.json'
    },
    'coco_2014_minival_ltps': {
        _IM_DIR:
            _DATA_DIR + '/shanghai/shanghai_ltps',
        _ANN_FN:
            _DATA_DIR + '/shanghai/shanghai_ltps/new.json'
    },
    'coco_2014_minival_ltps_all': {
        _IM_DIR:
            _DATA_DIR + '/shanghai/shanghai_ltps_all',
        _ANN_FN:
            _DATA_DIR + '/shanghai/shanghai_ltps_all/new.json'
    },
    'coco_2014_minival_mask': {
        _IM_DIR:
            _DATA_DIR + '/shanghai/shanghai_mask',
        _ANN_FN:
            _DATA_DIR + '/shanghai/shanghai_mask/new.json'
    },
    'coco_2014_minival_ltps_unlabel': {
        _IM_DIR:
            _DATA_DIR + '/shanghai/shanghai_ltps_unlabel',
        _ANN_FN:
            _DATA_DIR + '/shanghai/shanghai_ltps_unlabel/new.json'
    },
    'coco_2014_minival_ltps_unlabel1': {
        _IM_DIR:
            _DATA_DIR + '/shanghai/shanghai_ltps_unlabel1',
        _ANN_FN:
            _DATA_DIR + '/shanghai/shanghai_ltps_unlabel1/new.json'
    },
    'coco_2014_minival_tp': {
        _IM_DIR:
            _DATA_DIR + '/shanghai/shanghai_tp',
        _ANN_FN:
            _DATA_DIR + '/shanghai/shanghai_tp/new.json'
    },
    'coco_2014_minival_s6data1': {
        _IM_DIR:
            _DATA_DIR + '/S6data1',
        _ANN_FN:
            _DATA_DIR + '/S6data1/new.json'
    },
    'coco_2014_minival_s6data2': {
        _IM_DIR:
            _DATA_DIR + '/S6data2',
        _ANN_FN:
            _DATA_DIR + '/S6data2/new.json'
    },
    'coco_2014_minival_s6data2addtest': {
        _IM_DIR:
            _DATA_DIR + '/S6data2addtest',
        _ANN_FN:
            _DATA_DIR + '/S6data2addtest/new.json'
    },
    'coco_2014_minival_s6data3': {
        _IM_DIR:
            _DATA_DIR + '/S6data3',
        _ANN_FN:
            _DATA_DIR + '/S6data3/new.json'
    },
    'coco_2014_minival_s6data4': {
        _IM_DIR:
            _DATA_DIR + '/S6data4',
        _ANN_FN:
            _DATA_DIR + '/S6data4/new.json'
    },
    'coco_2014_minival_xiongmaotest_multi': {
        _IM_DIR:
            _DATA_DIR + '/xiongmaotest_multi',
        _ANN_FN:
            _DATA_DIR + '/xiongmaotest_multi/new.json'
    },
    'coco_2014_minival_xiongmaotest': {
        _IM_DIR:
            _DATA_DIR + '/xiongmaotest',
        _ANN_FN:
            _DATA_DIR + '/xiongmaotest/new.json'
    },
    'coco_2014_minival_44': {
        _IM_DIR:
            _DATA_DIR + '/A1_S1',
        _ANN_FN:
            _DATA_DIR + '/A1_S1/new.json'
    },
    'coco_2014_minival_5': {
        _IM_DIR:
            _DATA_DIR + '/S5',
        _ANN_FN:
            _DATA_DIR + '/S5/new.json'
    },
    'coco_2014_valminusminival': {
        _IM_DIR:
            _DATA_DIR + '/A1',
        _ANN_FN:
            _DATA_DIR + '/A1/new.json'
    },
    'coco_2015_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'coco_2015_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'coco_2017_test': {  # 2017 test uses 2015 test images
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json',
        _IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_2017_test-dev': {  # 2017 test-dev uses 2015 test images
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
        _IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_stuff_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_train2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/coco_stuff_train.json'
    },
    'coco_stuff_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/coco_stuff_val.json'
    },
    'keypoints_coco_2014_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_train2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2014.json'
    },
    'keypoints_coco_2014_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2014.json'
    },
    'keypoints_coco_2014_minival': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_minival2014.json'
    },
    'keypoints_coco_2014_valminusminival': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_valminusminival2014.json'
    },
    'keypoints_coco_2015_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'keypoints_coco_2015_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'voc_2007_train': {
        _IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_train.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2007_val': {
        _IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_val.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2007_test': {
        _IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_test.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2012_train': {
        _IM_DIR:
            _DATA_DIR + '/VOC2012/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/VOC2012/annotations/voc_2012_train.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/VOC2012/VOCdevkit2012'
    },
    'voc_2012_val': {
        _IM_DIR:
            _DATA_DIR + '/VOC2012/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/VOC2012/annotations/voc_2012_val.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/VOC2012/VOCdevkit2012'
    }
}


def datasets():
    """Retrieve the list of available dataset names."""
    return _DATASETS.keys()


def contains(name):
    """Determine if the dataset is in the catalog."""
    return name in _DATASETS.keys()


def get_im_dir(name):
    """Retrieve the image directory for the dataset."""
    return _DATASETS[name][_IM_DIR]


def get_ann_fn(name):
    """Retrieve the annotation file for the dataset."""
    return _DATASETS[name][_ANN_FN]


def get_im_prefix(name):
    """Retrieve the image prefix for the dataset."""
    return _DATASETS[name][_IM_PREFIX] if _IM_PREFIX in _DATASETS[name] else ''


def get_devkit_dir(name):
    """Retrieve the devkit dir for the dataset."""
    return _DATASETS[name][_DEVKIT_DIR]


def get_raw_dir(name):
    """Retrieve the raw dir for the dataset."""
    return _DATASETS[name][_RAW_DIR]
