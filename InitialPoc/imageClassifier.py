"""IMAGE CLASSIFIER"""

import numpy as np
import tensorflow as tf
import imageRGBtoGrayScale_lib as iR2G

def main():


    tf.logging.set_verbosity(tf.logging.WARN)
     # Path of the images
    img_path = "/disk1/phase2/data/train"

    img_path_test = "/disk1/phase2/data/test"
    images = []
    labels = []
    # Resolution of the images
    res1 = 28
    res2 = 28

    # Extracting images and labels
    labels, images = iR2G.listDataset(img_path, res1,res2)
    img_shape = (res1, res2)


    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(images)},
        y=np.array(labels),
        num_epochs=None,
        shuffle=True)
    # specifiying the features_set as individual image


    feature_x = tf.feature_column.numeric_column("x", shape=img_shape)

    feature_columns = [feature_x]


    hidden_layers = [784, 394, 196]
    # Construct Neural network classifier
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=hidden_layers, n_classes = 2, model_dir = "/disk1/phase2/Bharath/models")

    print("Training the classifier:")
    classifier.train(input_fn=train_input_fn, steps=1)


    test_labels, test_images = iR2G.listDataset(img_path_test,res1,res2)

 

    print ("########################   Testing the classifier:################################")

    print ("INPUT IMAGE1 :  /disk1/phase2/Bharath/test1.jpg")
    print ("INPUT IMAGE2 :  /disk1/phase2/Bharath/test2.jpg")

    print ("Kindly replace the above images with the images you want to test,. Please make sure to name them as test1 and test2")
    print ("The output prediction will be an array whose label is mentioned in the top eg: cat loading... label 0  In this cat has label 0 ")
    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_images)},
        y=np.array(test_labels),
        num_epochs=1,
        shuffle=False)


    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


    # Testing on single image
    test_image1 = "/disk1/phase2/Bharath/test1.jpg"
    test_image2 = "/disk1/phase2/Bharath/test2.jpg"

    # Convert the images to grayscale and 28 X 28 resolution
    sample_image1 = iR2G.img2gs_res(test_image1,28,28)
    sample_image2 = iR2G.img2gs_res(test_image2,28,28)


    # Classify two new flower samples.
    new_samples = np.array([sample_image1, sample_image2], dtype=np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": new_samples},
        num_epochs=1,
        shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]
    print(
    "New Samples, Class Predictions:    {}\n"
    .format(predicted_classes))

    # Exporting the model
    #export_dir = classifier.export_savedmodel( export_dir_base="/disk1/phase2/Bharath/models",serving_input_receiver_fn=serving_input_receiver_fn)

if __name__ == '__main__':
    main()
