#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import json
import os
import os.path as osp
import sys
import shutil   # to remove original output directory

import numpy as np
import PIL.Image

import labelme


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_dir', help='input annotated directory')
    parser.add_argument('output_dir', help='output dataset directory')
    # to overwrite output directory 
    parser.add_argument('--overwrite', help='overwrite output directory', action='store_true')
    args = parser.parse_args()
    '''
    if osp.exists(args.output_dir):
        if args.overwrite == True:
            print('Delete the ouput directory:', args.output_dir)
            shutil.rmtree(args.output_dir)
        else:
            print('Output directory already exists:', args.output_dir)
            sys.exit(1)
    os.makedirs(args.output_dir)
    '''
    class_names = []
    class_name_to_id = {}

    class_names.append('__ignore__')
    class_names.append('_background_')


    idx = 1
    for label_file in glob.glob(osp.join(args.input_dir, '**/*.json'), recursive=True):
        print('Generating dataset #{} from: {}'.format(idx,label_file))
        idx += 1

        with open(label_file) as f:
            data = json.load(f)

            # To check if the class name exist!! 
            wrong = False
            for shape in data['shapes']:
                cur_label = shape['label']
                if cur_label not in class_name_to_id.keys():
                    class_name_to_id[cur_label] = 1
                    class_names.append(cur_label)
                    wrong = True

    class_names = tuple(class_names)
    print('class_names:', class_names)
    out_class_names_file = osp.join(args.output_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)



if __name__ == '__main__':
    main()
