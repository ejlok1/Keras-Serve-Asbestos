# Deploy an Asbestos Rooftop detector on the web 


## Overview
The original project is based [this](https://github.com/ejlok1/Keras-Serve-Digit) which I extended on after a successful prototype. Basically I'm deploying the Asbestos Rooftop Detector that is built on the  Deep Learning model architecture, using the [tensorflow](https://github.com/tensorflow/tensorflow) framework. We wrap the whole model into a Webapp using the [Flask](http://flask.pocoo.org/) Micro Framework. And in GCP i'm using ComputeEngine which is easy to deploy even without using a container (Docker)

After testing in locally, we go to Google Cloud, activate the ComputeEngine App and host / serve up our solution! The website is [here](https://keras-serve-asbestos.appspot.com)


## Dependencies

```
mkdir Asbestos-Detector
conda create --name Asbestos-Detector python=3.5 pip numpy
activate Asbestos-Detector
pip install -r requirements.txt
```

## Testing locally

Once dependencies are installed, test to make sure it works on your localhost before going to Google Cloud. In command line:

```
cd Asbestos-Detector
activate Asbestos-Detector
python app.py
```

It's serving a saved Keras model to you via Flask. On the web browser the address is 
```http://localhost:5000)``` 
thou you need to change it on the app.py file. 

## Google Cloud hosting 
Select the Compute Engine solution, implement our model on an environement using the Google terminal. Here's a quick [guide](https://cloud.google.com/appengine/docs/flexible/python/quickstart) Then deploy! The live [website](https://keras-serve-asbestos.appspot.com) to load the file, and get the [outputs](https://keras-serve-asbestos.appspot.com/gallery) 


## Credits
Thank you [@ibrahimokdadov](https://github.com/ibrahimokdadov/upload_file_python) for the great videos on how to deal with images using Flask.  

## Further improvements 
- Documentation of the Google Cloud hosting process 
- Allow for more image types like .png or .JPEG etc 
- Provide another solution that links directly to google images API
- Enable app to work on tablets and smartphones 
- Get it to work on Dockers (and gRPC) 
- Get it to work on Kubeflow 
