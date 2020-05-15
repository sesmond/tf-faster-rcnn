# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

__sets = {}


def init_voc():
    from datasets.pascal_voc import pascal_voc
    # Set up voc_<year>_<split>
    for year in ['2007', '2012']:
        for split_cls in ['train', 'val', 'trainval', 'test']:
            name = 'voc_{}_{}'.format(year, split_cls)
            __sets[name] = (lambda split=split_cls, year=year: pascal_voc(split, year))

    for year in ['2007', '2012']:
        for split_cls in ['train', 'val', 'trainval', 'test']:
            name = 'voc_{}_{}_diff'.format(year, split_cls)
            __sets[name] = (lambda split=split_cls, year=year: pascal_voc(split, year, use_diff=True))


def init_coco():
    from datasets.coco import coco
    # Set up coco_2014_<split>
    for year in ['2014']:
        for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
            name = 'coco_{}_{}'.format(year, split)
            __sets[name] = (lambda split=split, year=year: coco(split, year))

    # Set up coco_2015_<split>
    for year in ['2015']:
        for split in ['test', 'test-dev']:
            name = 'coco_{}_{}'.format(year, split)
            __sets[name] = (lambda split=split, year=year: coco(split, year))


def init_gen():
    print("初始化数据")
    from datasets.general_data import general
    for split in ['train', 'test']:
        name = 'gen_{}'.format(split)
        # lambda 只是定义了方法，不在这里调用
        __sets[name] = (lambda split=split: general(split))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    # 把之前定义好的几个路径都执行一遍
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())


# 加载的时候就初始化数据结构
init_gen()

if __name__ == '__main__':
    print("hello")
    imdb = get_imdb("gen_train")
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method('gt')
    print('Set proposal method: {:s}'.format("gt"))
    # roidb = get_training_roidb(imdb)
    import roi_data_layer.roidb as rdl_roidb
    print('Preparing training data...')
    rdl_roidb.prepare_roidb(imdb)
    print('done')

    from utils.visualization import draw_bounding_boxes

    roidb = imdb.roidb
    print("sds")
    import  cv2
    for im_info in roidb:

        new_img = draw_bounding_boxes()
        cv2.imwrite("data/table/" ,new_img )
    """
    [{'boxes': array([[1079,  442, 1497,  517],
       [1101,  476, 1499,  549],
       [1171,  532, 1497,  642]], dtype=uint16), 
       'gt_classes': array([1, 1, 1], dtype=int32), 
       'gt_overlaps': <3x2 sparse matrix of type '<class 'numpy.float32'>'
	with 3 stored elements in Compressed Sparse Row format>, 
	'flipped': False, 
	'seg_areas': array([31844., 29526., 36297.], dtype=float32), 
	'image': '/1.jpg',
	 'width': 1463, 'height': 1198, 
	 'max_classes': array([1, 1, 1]), 
	 'max_overlaps': array([1., 1., 1.], dtype=float32)}]
    """