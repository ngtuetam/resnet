import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add
from tensorflow.keras import Model

def basic_block(input, filter_num, stride=1,stage_idx=-1, block_idx=-1):
  '''BasicBlock use stack of two 3x3 convolutions layers

  Args:
    filter_num: the number of filters in the convolution
    stride: the number of strides in the convolution. stride = 1 if you want
            output shape is same as input shape
    stage_idx: index of current stage
    block_idx: index of current block in stage
  '''
  # conv3x3
  conv1=Conv2D(filters=filter_num,
               kernel_size=3,
               strides=stride,
               padding='same',
               kernel_initializer='he_normal',
               name='conv{}_block{}_1_conv'.format(stage_idx, block_idx))(input)
  bn1=BatchNormalization(name='conv{}_block{}_1_bn'.format(stage_idx, block_idx))(conv1)
  relu1=ReLU(name='conv{}_block{}_1_relu'.format(stage_idx, block_idx))(bn1)
  # conv3x3
  conv2=Conv2D(filters=filter_num,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               name='conv{}_block{}_2_conv'.format(stage_idx, block_idx))(relu1)
  bn2=BatchNormalization(name='conv{}_block{}_2_bn'.format(stage_idx, block_idx))(conv2)

  return bn2

def bottleneck_block(input, filter_num, stride=1, stage_idx=-1, block_idx=-1):
  '''BottleNeckBlock use stack of 3 layers: 1x1, 3x3 and 1x1 convolutions

  Args:
    filter_num: the number of filters in the convolution
    stride: the number of strides in the convolution. stride = 1 if you want
            output shape is same as input shape
    stage_idx: index of current stage
    block_idx: index of current block in stage
  '''
  # conv1x1
  '''
  Initializers define cach dat weight ngau nhien ban dau cua cac lop keras
  kernel_initializer: stddev = sqrt(2 / fan_in)
  - draws samples from a truncated normal distribution centered on 0
  - fan_in: the number of input units in the weight tensor
  '''
  conv1=Conv2D(filters=filter_num,
               kernel_size=1,
               strides=stride,
               padding='valid',   
               kernel_initializer='he_normal',
               name='conv{}_block{}_1_conv'.format(stage_idx, block_idx))(input)
  bn1=BatchNormalization(name='conv{}_block{}_1_bn'.format(stage_idx, block_idx))(conv1)
  relu1=ReLU(name='conv{}_block{}_1_relu'.format(stage_idx, block_idx))(bn1)
  # conv3x3
  conv2=Conv2D(filters=filter_num,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               name='conv{}_block{}_2_conv'.format(stage_idx, block_idx))(relu1)
  bn2=BatchNormalization(name='conv{}_block{}_2_bn'.format(stage_idx, block_idx))(conv2)
  relu2=ReLU(name='conv{}_block{}_2_relu'.format(stage_idx, block_idx))(bn2)
  # conv1x1
  conv3=Conv2D(filters=4*filter_num,
               kernel_size=1,
               strides=1,
               padding='valid',
               kernel_initializer='he_normal',
               name='conv{}_block{}_3_conv'.format(stage_idx, block_idx))(relu2)
  bn3=BatchNormalization(name='conv{}_block{}_3_bn'.format(stage_idx, block_idx))(conv3)
  
  return bn3

def resblock(input, filter_num, stride=1, use_bottleneck=False,stage_idx=-1, block_idx=-1):
  '''A complete `Residual Unit` of ResNet

  Args:
    filter_num: the number of filters in the convolution
    stride: the number of strides in the convolution. stride = 1 if you want
            output shape is same as input shape
    use_bottleneck: type of block: basic or bottleneck
    stage_idx: index of current stage
    block_idx: index of current block in stage
  '''
  if use_bottleneck:
    residual = bottleneck_block(input, filter_num, stride,stage_idx, block_idx)
    expansion=4
  else:
    residual = basic_block(input, filter_num, stride,stage_idx, block_idx)
    expansion=1

  shortcut=input
  # use projection short cut when dimensions increase
  if stride>1 or input.shape[3]!=residual.shape[3]:
    shortcut=Conv2D(expansion*filter_num,
                    kernel_size=1,
                    strides=stride,
                    padding='valid',
                    kernel_initializer='he_normal',
                    name='conv{}_block{}_projection-shortcut_conv'.format(stage_idx, block_idx))(input)
    shortcut=BatchNormalization(name='conv{}_block{}_projection-shortcut_bn'.format(stage_idx, block_idx))(shortcut)

  output=Add(name='conv{}_block{}_add'.format(stage_idx, block_idx))([residual, shortcut])

  return ReLU(name='conv{}_block{}_relu'.format(stage_idx, block_idx))(output)

#   # conv1
#   net=Conv2D(filters=64,
#              kernel_size=7,
#              strides=2,
#              padding='same',
#              kernel_initializer='he_normal',
#              name='conv1_conv')(input)
#   net=BatchNormalization(name='conv1_bn')(net)
#   net=ReLU(name='conv1_relu')(net)
#   net=MaxPooling2D(pool_size=3,
#                    strides=2,
#                    padding='same',
#                    name='conv1_max_pool')(net)

#   # conv2_x, conv3_x, conv4_x, conv5_x
#   filters=[64,128,256,512]
#   for i in range(len(filters)):
#     net=stage(input = net,
#               filter_num = filters[i],
#               num_block = layers[i],
#               use_downsample = i!=0,
#               use_bottleneck = use_bottleneck,
#               stage_idx = i+2)
