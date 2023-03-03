#-*-coding:utf8-*-#

from PIL import Image
from pylab import * 
import numpy as np
import glob
import sys
caffe_root = "/home/gpj/caffe-master/python"
sys.path.append(caffe_root)
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
import pickle
import cv2
import shutil
import os
import time


def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3) and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 
               + ((0, 0),) * (data.ndim - 3))  
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.show()
    plt.axis('off')

#Initialize transformers

def initialize_transformer(image_mean, is_flow):
  shape = (10*16, 3, 227, 227)  
  transformer = caffe.io.Transformer({'data': shape})
  channel_mean = np.zeros((3,227,227))
  for channel_index, mean_val in enumerate(image_mean):
    channel_mean[channel_index, ...] = mean_val
  transformer.set_mean('data', channel_mean)
  transformer.set_raw_scale('data', 255)
  transformer.set_channel_swap('data', (2, 1, 0))
  transformer.set_transpose('data', (2, 0, 1))
  transformer.set_is_flow('data', is_flow)
  return transformer
ucf_mean_RGB = np.zeros((3,1,1))
ucf_mean_flow = np.zeros((3,1,1))
ucf_mean_flow[:,:,:] = 128
ucf_mean_RGB[0,:,:] = 90.013
ucf_mean_RGB[1,:,:] = 97.029
ucf_mean_RGB[2,:,:] = 101.396

transformer_RGB = initialize_transformer(ucf_mean_RGB, False)
transformer_flow = initialize_transformer(ucf_mean_flow,True)


def LRCN_classify_video(frames, net, transformer, is_flow):
  clip_length = 16
  offset =10        #8
  input_images = []
  for im in frames:
    input_im = caffe.io.load_image(im)
    input_im=cv2.resize(input_im,(256,256),interpolation=cv2.INTER_AREA)
    if (input_im.shape[0] < 256):
      input_im = caffe.io.resize_image(input_im, (256,256))
    input_images.append(input_im)
  vid_length = len(input_images)
  input_data = []
  for i in range(0,vid_length,offset):
    if (i + clip_length) < vid_length:
      input_data.extend(input_images[i:i+clip_length])
    else:  
      input_data.extend(input_images[-clip_length:])
  output_predictions = np.zeros((len(input_data),101))
  for i in range(0,len(input_data),clip_length):   
    clip_input = input_data[i:i+clip_length]
    clip_input = caffe.io.oversample(clip_input,[227,227])
    clip_clip_markers = np.ones((clip_input.shape[0],1,1,1))
    clip_clip_markers[0:10,:,:,:] = 0
    caffe_in = np.zeros(np.array(clip_input.shape)[[0,3,1,2]], dtype=np.float32)
    for ix, inputs in enumerate(clip_input):         
      caffe_in[ix] = transformer.preprocess('data',inputs)
    out = net.forward_all(data=caffe_in, clip_markers=np.array(clip_clip_markers))
    output_predictions[i:i+clip_length] = np.mean(out['probs'],1)  
    #print output_predictions

  return np.mean(output_predictions,0).argmax(), output_predictions
#Models and weights
def STA_LSTM_classify_video(frames, net, transformer,all_videos,now_test_vodeo,now_acc):
  clip_length = 16   
  offset = 10
  input_images = []
  display=1
  for im in frames:
    input_im = caffe.io.load_image(im)
    if (input_im.shape[0] < 256):
      input_im = caffe.io.resize_image(input_im, (256, 256))
    input_images.append(input_im)
  vid_length = len(input_images)
  print "LRCN_vid_length",vid_length
  input_data = []
  for i in range(0, vid_length, offset):
    if (i + clip_length) < vid_length:
      input_data.extend(input_images[i:i + clip_length])
    else:  # video may not be divisible by clip_length
      input_data.extend(input_images[-clip_length:])
  output_predictions = np.zeros((len(input_data), 101))
  t = time.time()
  print "len(input_data)",len(input_data)
  for i in range(0, len(input_data), clip_length):
    if display ==1:
      print 'Now on %d/%d .' % (all_videos, now_test_vodeo) 
      if now_test_vodeo-1>=1:
        print 'Now acc: %.4f \n' % (float(now_acc)/float(now_test_vodeo-1))
      display =0

    clip_input = input_data[i:i + clip_length]
    clip_input = caffe.io.oversample(clip_input, [227, 227])
    clip_input_next = input_data[i+2:i+2+clip_length]
    clip_input_next = caffe.io.oversample(clip_input_next, [227, 227])
    clip_clip_markers = np.ones((clip_input.shape[0], 1, 1, 1))
    clip_clip_markers[0:10, :, :, :] = 0
    caffe_in = np.zeros(np.array(clip_input.shape)[[0, 3, 1, 2]], dtype=np.float32)
    caffe_in_next = np.zeros(np.array(clip_input_next.shape)[[0, 3, 1, 2]], dtype=np.float32)
    for ix, inputs in enumerate(clip_input):
      caffe_in[ix] = transformer.preprocess('data', inputs)
    for ix_next, inputs_next in enumerate(clip_input_next):
      caffe_in_next[ix_next] = transformer.preprocess('data', inputs_next)
    out = net.forward_all(data=caffe_in,data_next=caffe_in_next,clip_markers=np.array(clip_clip_markers))
    output_predictions[i:i + clip_length] = np.mean(out['probs'], 1)  

  

  return np.mean(output_predictions, 0).argmax(), output_predictions
  # Models and weights

def LSTM_network_videos_detector(net,model,txt):
  correct_number = 0                                 
  RGB_lstm_orig = model                                
  lstm_model = net                                       
  test_txt = txt                                        
  video_labels = open(test_txt).readlines()
  videos = [v.split(' ')[0] for v in video_labels]       
  labels = [int(v.split(' ')[1]) for v in video_labels]  
  detection_fail = []                                 
  detection_succ_label = []                              
  all_videos = len(videos)

 
  
  acc_count=0

  for i in range(0, all_videos):
    print videos[i]
    RGB_frames = glob.glob('%s%s/*.jpg' % ('../',videos[i]))  
    print RGB_frames
    
    RGB_lstm_net = caffe.Net(lstm_model, RGB_lstm_orig, caffe.TEST)
    #class_RGB_LRCN, predictions_RGB_LRCN_orig = LRCN_classify_video(RGB_frames, RGB_lstm_net, transformer_RGB, False)           
    class_RGB_LRCN, predictions_RGB_LRCN_orig = STA_LSTM_classify_video(RGB_frames, RGB_lstm_net, transformer_RGB,all_videos,i+1,acc_count)      
    del RGB_lstm_net

    if class_RGB_LRCN == labels[i:i + 1]:                
       detection_succ_label.append(labels[i])             
       correct_number = correct_number + 1           
    else:                                                 
      detection_fail.append(videos[i])
    acc_count=correct_number  

  print "test_over.\n"
  
  

  print "all_videos", all_videos

  print "correct_number ", correct_number

  myset = set(detection_succ_label)  
  action_hash = pickle.load(open('action_hash_rev.p', 'rb'))
  for item in myset:
    print("the %s accuracy is %f" % (action_hash[item], float(detection_succ_label.count(item)) / float(labels.count(item)))) 

  print 'Total accuracy is %f.\n' % (float(correct_number) / float(all_videos))

 

  print "detect_fail", detection_fail


if __name__ =='__main__':

    lstm_net = 'SiamMAST.prototxt'

    RGB_lstm_orig = '../iter_10000.caffemodel'  

    test_txt = "paper_test.txt"

    LSTM_network_videos_detector(lstm_net,RGB_lstm_orig,test_txt)