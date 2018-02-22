import numpy as np
import tensorflow as tf
import imageRGBtoGrayScale_lib as iR2G


tf.logging.set_verbosity(tf.logging.INFO)


def main():


    img_path = "/home/biarca/disk1/phase2/Bharath/workspace/MyTest/dog_and_cat/train"

    img_path_test = "/home/biarca/disk1/phase2/Bharath/workspace/MyTest/dog_and_cat/test"

    dir_path = "/home/biarca/disk1/phase2/Bharath/workspace/MyTest/dog_and_cat/output_model/"
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
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=hidden_layers, n_classes = 2, model_dir = "/home/biarca/disk1/phase2/Bharath/workspace/MyTest/dog_and_cat/models")

    print("Training the classifier:")
    classifier.train(input_fn=train_input_fn, steps=10000)


    test_labels, test_images = iR2G.listDataset(img_path_test,res1,res2)

 

    '''print ("########################   Testing the classifier:################################")

    print ("INPUT IMAGE1 :  /home/biarca/disk1/phase2/Bharath/test1.jpg")
    print ("INPUT IMAGE2 :  /home/biarca/disk1/phase2/Bharath/test2.jpg")

    print ("Kindly replace the above images with the images you want to test,. Please make sure to name them as test1 and test2")
    print ("The output prediction will be an array whose label is mentioned in the top eg: cat loading... label 0  In this cat has label 0 ")'''
    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_images)},
        y=np.array(test_labels),
        num_epochs=1,
        shuffle=False)


    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    # Export the model

    # # Adding the signature_def 
    #feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    feature_spec = { "x" : tf.FixedLenFeature(dtype=tf.float32, shape=[784])}



    export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    servable_model_dir = "output_models/"
    servable_model_path = classifier.export_savedmodel(servable_model_dir, export_input_fn)
    print "Model Exported  to output_models folder"

if __name__ == '__main__':
    main()

