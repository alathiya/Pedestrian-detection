import argparse
import cv2
import numpy as np

from inference import Network



def main():
    args = get_args()
    perform_inference(args)


def get_args():
    '''
    Gets the arguments from the command line.
    '''

    parser = argparse.ArgumentParser("Basic Edge App with Inference Engine")

    c_desc = "CPU extension file location, if applicable"
    d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    i_desc = "The location of the input image"
    m_desc = "The location of the model XML file"
    
    parser._action_groups.pop()
    
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument("-i", help=i_desc, required=True)
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-c", help=c_desc, default=None)
    optional.add_argument("-d", help=d_desc, default="CPU")
    
    args = parser.parse_args()

    return args

def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''

    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image

def processed_and_create_output(output, input_img, preprocess_input_img, input_img_name):

    y_ = preprocess_input_img.shape[2]
    x_ = preprocess_input_img.shape[3]

    targetSize_y = input_img.shape[0]
    targetSize_x = input_img.shape[1]
    
    x_scale = targetSize_x / x_
    y_scale = targetSize_y / y_

    
    output = output['detection_out'].squeeze()

    box_lst = []

    for item in range(len(output[0])):
        box = output[item]

        if box[2] > 0.8:
            #box_lst.append(box)
            xmin = int(box[3] * x_ * x_scale)
            ymin = int(box[4] * y_ *  y_scale)
            xmax = int(box[5] * x_ * x_scale) 
            ymax = int(box[6] * y_ * y_scale)

            cv2.rectangle(input_img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
    
    # Save down the resulting image
    output_image_file_name = input_img_name.split('/')[1].split('.')[0] + "_output.png"
    path = 'output_images/' + output_image_file_name
    #print(path)
    cv2.imwrite(path,input_img)


def perform_inference(args):
    '''
    Performs inference on an input image, given a model.
    '''
    
    # Create a Network for using the Inference Engine
    inference_network = Network()
    
    # Load the model in the network, and obtain its input shape
    n, c, h, w = inference_network.load_model(args.m, args.d, args.c)

     # Read the input image
    image = cv2.imread(args.i)
    
    # Preprocess the input image
    preprocessed_image = preprocessing(image, h, w)

    # Perform synchronous inference on the image
    inference_network.sync_inference(preprocessed_image)

    # Obtain the output of the inference request
    output = inference_network.extract_output()

    # process output from inference to extract and display bounding boxes around detected pedestrian. 
    # Create an output image based on network
    processed_and_create_output(output, image, preprocessed_image, args.i)
    
    

if __name__ == "__main__":
    main()