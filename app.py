import os
from flask import Flask, request, render_template, send_from_directory
import sys
#tell our app where our saved model is
#sys.path.append(os.path.abspath("./object_detection"))
sys.path.append(os.path.abspath("."))
from object_detection.Object_detector import * 
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    # folder_name = request.form['superhero']
    '''
    # this is to verify that folder to upload to exists.
    if os.path.isdir(os.path.join(APP_ROOT, 'files/{}'.format(folder_name))):
        print("folder exist")
    '''
    #target = os.path.join(APP_ROOT, 'files/{}'.format(folder_name))
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    
    if not os.path.isdir(target):
        os.mkdir(target)
        
    # print(request.files.getlist("file"))
    
    for upload in request.files.getlist("file"):
        #print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        # This is to verify files are supported
        ext = os.path.splitext(filename)[1]
        if (ext == ".jpg"): # or (ext == ".png"):
            print("File supported moving on...")
        else:
            render_template("Error.html", message="Files uploaded are not supported...")
        destination = "/".join([target, filename])
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        upload.save(destination)

        #################
        # Run model
        #################
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_FROZEN_GRAPH =  'frozen_inference_graph.pb'
        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')
        NUM_CLASSES = 1
        IMAGE_SIZE = (12, 8)

        # Load a (frozen) Tensorflow model into memory
        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
        # Loading label map
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        PATH_TO_TEST_IMAGES_DIR = 'images'
        PATH_TO_OUTPUT_IMAGES_DIR = 'static'
        #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Picture{}.jpg'.format(i)) for i in range(134, 136) ]
        
        image_path = os.path.join(PATH_TO_TEST_IMAGES_DIR,filename)
        image = Image.open(image_path)
        image_np = load_image_into_numpy_array(image)
        
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              output_dict['detection_boxes'],
              output_dict['detection_classes'],
              output_dict['detection_scores'],
              category_index,
              instance_masks=output_dict.get('detection_masks'),
              use_normalized_coordinates=True,
              line_thickness=8)
        plt.imsave(os.path.join(PATH_TO_OUTPUT_IMAGES_DIR,filename),image_np)
        

    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("complete.html", image_name=filename)


#@app.route('/upload/<filename>')
#def send_image(filename):
#    return send_from_directory("images", filename)
    
@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("static", filename)

@app.route('/gallery')
def get_gallery():
    image_names = os.listdir('./static')
    print(image_names)
    return render_template("outputs.html", image_names=image_names)



if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='localhost', port=port)  #app.run(host='0.0.0.0', port=port) #app.run(host='localhost', port=port)  
	#optional if we want to run in debugging mode
	#app.run(debug=True)

