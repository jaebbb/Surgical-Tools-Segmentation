import argparse
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
import cv2

from mypath import Path
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.metrics import Evaluator
from utils.util import plot_confusion_matrix
# from apex import amp
import torch

import fcn
from torchvision import transforms
from dataloaders import custom_transforms as tr


from PIL import Image
# cuda check 할것...
import random
import warnings
import time
from tqdm import tqdm
warnings.filterwarnings(action='ignore')


def parse_cfg(cfgfile):

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')     #store the lines in a list
    lines = [x for x in lines if len(x) > 0] #get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']  
    lines = [x.rstrip().lstrip() for x in lines]

    
    block = {}
    
    for line in lines:
        if line[0] == "[":               #This marks the start of a new block
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    
    return block



class Deeplab(object):
    def __init__(self, cfgfile):
        self.args = parse_cfg(cfgfile)

        self.nclass =int(self.args['nclass'])

        model = DeepLab(num_classes=self.nclass,
                        backbone=self.args['backbone'],
                        output_stride=int(self.args['out_stride']),
                        sync_bn=bool(self.args['sync_bn']),
                        freeze_bn=bool(self.args['freeze_bn']))

        weight = None

        self.criterion = SegmentationLosses(weight=weight, cuda= True).build_loss(mode=self.args['loss_type'])

        self.model = model
        self.evaluator = Evaluator(self.nclass)

        # Using cuda

        self.model = self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=[0])
        patch_replication_callback(self.model)
        self.resume = self.args['resume']

        # Resuming checkpoint
        if self.resume is not None:
            if not os.path.isfile(self.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(self.resume))
            checkpoint = torch.load(self.resume)

            self.model.module.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(self.resume, checkpoint['epoch']))


    def untransform(self, img, lbl=None):
        mean_bgr = np.array([0.485, 0.456, 0.406])
        std_bgr = np.array([0.229, 0.224, 0.225])

        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img *= std_bgr
        img += mean_bgr
        img *= 255
        img = img.astype(np.uint8)

        lbl = lbl.numpy()
        return img, lbl

    def transform(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(size=513),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)
    

    def validation(self, src, data, ClassName, tar=None):

        self.model.eval()
        self.evaluator.reset()
        test_loss = 0.0

        w,h = src.size
        sample = {'image': src, 'label': tar}
        sample = self.transform(sample)
        image , target = sample['image'], sample['label']

        
        # for the dimension
        image = torch.unsqueeze(image, 0)
        target = torch.unsqueeze(target,0)
        image, target = image.cuda(), target.cuda()



        with torch.no_grad():
            output = self.model(image)


        imgs = image.data.cpu()
        lbl_pred = output.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu() #cpu()

        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):

            img, lt = self.untransform(img, lt)


            viz = fcn.utils.visualize_segmentation(
                lbl_pred=lp, lbl_true=None, img=img, n_class=10,
                label_names = [' ' , '11_pforceps','12_mbforceps','13_mcscissors','15_pcapplier','18_pclip','20_sxir','19_mtclip','17_mtcapplier','14_graspers'])
          

            width = int(viz.shape[1] / 3 * 2)
            #Image.fromarray(viz[:,width:,:]).save(str(data)+'.jpg')
            def hide(plt):
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)

            plt.figure(figsize=(20,10))
            plt.subplot(2,2,1)
            plt.imshow(src)
            plt.title('Image')
            hide(plt)

            plt.subplot(2,2,2)
            plt.imshow(tar)
            plt.title('SegmentationClass')
            hide(plt)

            plt.subplot(2,2,3)
            plt.imshow(Image.open('./dataset/output/SegmentationClassVisualization/'+data+'.jpg'))
            plt.title('SegmentationVisuallization')
            hide(plt)
            
            plt.subplot(2,2,4)
            plt.imshow(cv2.resize(np.float32(Image.fromarray(viz[:,width:,:]))/255, (w,h), interpolation=cv2.INTER_LINEAR ))
            plt.title('Prediction')
            hide(plt)
            if not os.path.isdir('result'):
                os.makedirs('result')
            plt.savefig('result/%s.png'%(data))

        
        if tar is not None:
            loss = self.criterion(output, target)
            test_loss += loss.item()
            #tbar.set_description('Test loss: %.3f' % (test_loss))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target[:,145:815,:], pred[:,145:815,:])
           
        
            # Fast test during the training
            Acc = self.evaluator.Pixel_Accuracy()
            Acc_class = self.evaluator.Pixel_Accuracy_Class()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            print('Validation:')
            #print('[Epoch: %d, numImages: %5d]' % (epoch, self.args.batch_size + image.data.shape[0]))
            print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
            print('Loss: %.3f' % test_loss)


            plot_confusion_matrix(self.evaluator.confusion_matrix, ClassName)
            plt.savefig('result/%s_CM.png'%(data))


        #return viz[:,width:, :]




    def validation_matrix(self, src, data, tar):

        self.model.eval()
        #self.evaluator.reset()
        sample = {'image': src, 'label': tar}
        sample = self.transform(sample)
        image , target = sample['image'], sample['label']
        # for the dimension
        image = torch.unsqueeze(image, 0)
        target = torch.unsqueeze(target,0)
        image = image.cuda()
    

        with torch.no_grad():
            output = self.model(image)

        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        # Add batch sample into evaluator
        self.evaluator.add_batch(target[:,145:815,:], pred[:,145:815,:])


        #return self.evaluator.confusion_matrix
                



def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    # cuda, seed and logging
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    # evaluation option
    parser.add_argument('--num_image', type=int, default=1,
                        help='visualization number of image')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='data list shuffle')
    parser.add_argument('--result', type=str, default='eachimage',
                        choices=['eachimage', 'matrix'],
                        help='data list shuffle')

    args = parser.parse_args()

    # default settings for epochs, batch_size and lr

    print(args)
    deeplab = Deeplab('cfg/deeplab.cfg')


    with open('./dataset/val.txt','r') as f:
        datalist = []
        for data in f.readlines():
            if '\n' in data:
                data = data[:-1]
            datalist.append(data)

    with open('./dataset/output/class_names.txt','r') as f:
            ClassName = []
            for cl in f.readlines():
                if '\n' in cl:
                    cl = cl[:-1]
                ClassName.append(cl)
    
    if args.shuffle:
        random.shuffle(datalist)

    if args.result =='matrix':
        start = time.time()
        for data in tqdm(datalist):
            src = Image.open('./dataset/output/JPEGImages/'+data +'.jpg').convert('RGB')
            tar = Image.open('./dataset/output/SegmentationClassPNG/'+data+'.png')
            deeplab.validation_matrix(src, data, tar)
        cm = deeplab.evaluator.confusion_matrix
        end = time.time()
        plot_confusion_matrix(cm/len(datalist),ClassName)
        plt.savefig('result/all_CM.png')
        print('Image : %d | Time : %.2f | fps : %.1f'%(len(datalist), end - start, len(datalist)/(end-start)))


    elif args.result == 'eachimage':
        for i in range(args.num_image):
            data = datalist[i]
            src = Image.open('./dataset/output/JPEGImages/'+data +'.jpg').convert('RGB')
            tar = Image.open('./dataset/output/SegmentationClassPNG/'+data+'.png')
            deeplab.validation(src, data, ClassName, tar)

        
    
        

    #trainer.writer.close()

if __name__ == "__main__":
    main()