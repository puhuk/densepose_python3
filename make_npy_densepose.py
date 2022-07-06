import os
import glob
from PIL import Image
import pickle
import numpy as np
img_list = glob.glob('../deepfashion//Flow-Style-VTON/train/dataset/VITONHD_traindata/vitondHD/zalando-hd-resized/train/image/*.jpg')

os.environ['MKL_THREADING_LAYER'] = 'GNU'

for item in img_list:
    os.system('python tools/apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml \
    https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl ' \
    + item + ' --output tmp.pkl -v')

    name = item.split('/')[-1].split('.')[0]

    img = Image.open(item)
    img_w ,img_h = img.size
    with open('output.pkl','rb') as f:
        data=pickle.load(f)
    i = data[0]['pred_densepose'][0].labels.cpu().numpy()
    uv = data[0]['pred_densepose'][0].uv.cpu().numpy()
    iuv = np.stack((uv[1,:,:], uv[0,:,:], i * 0,))
    iuv = np.transpose(iuv, (1,2,0))

    box = data[0]["pred_boxes_XYXY"][0]
    box[2]=box[2]-box[0]
    box[3]=box[3]-box[1]
    x,y,w,h=[int(v) for v in box]
    bg=np.zeros((img_h,img_w,3))
    bg[y:y+h,x:x+w,:]=iuv

    np.save('../deepfashion//Flow-Style-VTON/train/dataset/VITONHD_traindata/vitondHD/zalando-hd-resized/train/densepose_npy/' + name + '.npy', bg)


img_list = glob.glob('../deepfashion//Flow-Style-VTON/train/dataset/VITONHD_traindata/vitondHD/zalando-hd-resized/test/image/*.jpg')

for item in img_list:
    os.system('python tools/apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml \
    https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl ' \
    + item + ' --output tmp.pkl -v')

    name = item.split('/')[-1].split('.')[0]

    img = Image.open(item)
    img_w ,img_h = img.size
    with open('output.pkl','rb') as f:
        data=pickle.load(f)
    i = data[0]['pred_densepose'][0].labels.cpu().numpy()
    uv = data[0]['pred_densepose'][0].uv.cpu().numpy()
    iuv = np.stack((uv[1,:,:], uv[0,:,:], i * 0,))
    iuv = np.transpose(iuv, (1,2,0))

    box = data[0]["pred_boxes_XYXY"][0]
    box[2]=box[2]-box[0]
    box[3]=box[3]-box[1]
    x,y,w,h=[int(v) for v in box]
    bg=np.zeros((img_h,img_w,3))
    bg[y:y+h,x:x+w,:]=iuv

    np.save('../deepfashion//Flow-Style-VTON/train/dataset/VITONHD_traindata/vitondHD/zalando-hd-resized/test/densepose_npy/' + name + '.npy', bg)


