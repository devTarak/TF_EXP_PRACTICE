from __future__ import print_function

# This is a placeholder for a Google-internal import.

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import regression_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import imageRGBtoGrayScale_lib as iR2G
import numpy as np
import json

tf.app.flags.DEFINE_string('server', '0.0.0.0:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('x', 'housetrain_adult_dog_hero.jpg', '')
FLAGS = tf.app.flags.FLAGS

test_image = "/home/biarca/disk1/phase2/Bharath/workspace/dog1.jpeg"


def main(_):





  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  # Send request

  # See prediction_service.proto for gRPC request/response details.

  # Loop throught the images in folder

  dir_path = path
  # Read the directories available
  list_files = os.listdir(path)
  labels = []
  images = []
  f = open("answer.csv", "w+")
  # Label each folder with a value starting
  # from zero
  for i in list_files:
     print i + " loading ..." 
     print i.split(".")
     f.write(i+"\n")

 


 
  sample_image1 = iR2G.img2gs_res(test_image,28,28)

  new_samples = np.array([sample_image1], dtype=np.float32)
  
  image = new_samples.reshape(784,)

  print(image.shape)







  # Feature_dict
  request = classification_pb2.ClassificationRequest()
  request.model_spec.name = 'cat_and_dog'
  request.model_spec.signature_name = 'serving_default'
  example = request.input.example_list.examples.add()
  example.features.feature['x'].float_list.value.extend(image.astype(float))
  result = stub.Classify(request, 10.0)  # 10 secs timeout
  
  
  label0_value = result.result.classifications[0].classes[0].score
  label1_value = result.result.classifications[0].classes[1].score


  if (label0_value > label1_value):
      print(0)
  else:
      print(1)




if __name__ == '__main__':
  tf.app.run()


