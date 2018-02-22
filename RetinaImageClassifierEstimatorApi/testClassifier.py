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

tf.app.flags.DEFINE_string('server', '192.168.2.153:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('x', 'housetrain_adult_dog_hero.jpg', '')
FLAGS = tf.app.flags.FLAGS

test_image = "/home/biarca/disk1/phase2/Tarak/stare-tensorflow-estimator/stare/train/diabeticRetinopahthy/im0009.jpg"


def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  # Send request

  # See prediction_service.proto for gRPC request/response details.

  sample_image1 = iR2G.img2gs_res(test_image,28,28)

  new_samples = np.array([sample_image1], dtype=np.float32)
  
  image = new_samples.reshape(784,)

  print(image.shape)

  #f = open(test_image, 'rb')
  #data = f.read()

  # Feature_dict
  request = classification_pb2.ClassificationRequest()
  request.model_spec.name = 'default'
  request.model_spec.signature_name = 'serving_default'
  example = request.input.example_list.examples.add()
  example.features.feature['x'].float_list.value.extend(image.astype(float))
  result = stub.Classify(request, 10.0)  # 10 secs timeout
  print(result)


if __name__ == '__main__':
  tf.app.run()


