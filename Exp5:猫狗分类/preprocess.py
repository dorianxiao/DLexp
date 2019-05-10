from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import os
import random



def int64_feature(values):
	if not isinstance(values, (tuple, list)):
		values = [values]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
  
def bytes_feature(values):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
	
def image_to_tfexample(image_data, image_format, height, width, class_id):
	return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))

# 定义图像读取对象
class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

# 根据图像文件所在目录返回猫、狗的包含绝对路径的文件名列表
def get_files(file_dir):
    cats = []
    dogs = []
    
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0]=='cat':
            cats.append(os.path.join(file_dir, file))
        else:
            dogs.append(os.path.join(file_dir, file))
    print('总共有 %d 只猫\n和 %d 只狗。' %(len(cats), len(dogs)))
    
    return cats, dogs

# 划分训练集和验证集	
def get_train_val(cats, dogs):
	random.shuffle(cats)
	random.shuffle(dogs)
	
	training_filenames = cats[VAL_SIZE:] + dogs[VAL_SIZE:]
	validation_filenames = cats[:VAL_SIZE] + dogs[:VAL_SIZE]
	return training_filenames, validation_filenames

# 将图像数据集预处理后写入tfrecord文件	
def convert_dataset(split_name, filenames, dataset_dir):
	assert split_name in ['train', 'validation']
	
	with tf.Graph().as_default():
		image_reader = ImageReader()

		with tf.Session('') as sess:
		
			tfrcd = 'dogsVScats_%s_*.tfrecord' % split_name
			output_filename = os.path.join(dataset_dir, tfrcd)

			with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
				for i in range(len(filenames)):
					# 读取文件:
					image_data = tf.gfile.GFile(filenames[i], 'rb').read()
					height, width = image_reader.read_image_dims(sess, image_data)

					class_name = os.path.basename(filenames[i]).split('.')[0]
					if class_name == 'cat':
						class_id = 0
					else:
						class_id = 1

					example = image_to_tfexample(
						image_data, b'JPEG', height, width, class_id)
					tfrecord_writer.write(example.SerializeToString())

if __name__ == "__main__":
	# 图像数据所在目录
	FILE_DIR = r'./data/train'
	# tfrecord文件保存目录
	SAVE_PATH = r'./data'
	# 验证集大小
	VAL_SIZE = 2500
	
	# 得到猫、狗文件名列表
	cats, dogs = get_files(FILE_DIR)
	# 划分训练集、验证集
	training, validation = get_train_val(cats, dogs)
	
	# 生成对应的tfrecord文件
	convert_dataset('train', training, SAVE_PATH)
	convert_dataset('validation', validation, SAVE_PATH)


	