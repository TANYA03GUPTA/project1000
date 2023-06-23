from flask import Flask, render_template, request


from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from PIL import Image


# from transformers import TFViTForImageClassification, create_optimizer


# num_train_steps = 4679*30
# optimizer, lr_schedule = create_optimizer(
#    init_lr=3e-5,
#    num_train_steps=num_train_steps,
#    weight_decay_rate=0.01,
#    num_warmup_steps=0,

# )


# model = load_model(
#   'C:/Users/lenovo/Downloads/vit_ln_data_7_to_15_new.hdf5', custom_objects={'AdamWeightDecay': optimizer})


app = Flask(__name__)
#model = VGG16()
model = load_model(
    'MobileNet_7_15.hdf5')


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']

    image_path = "images\\" + imagefile.filename

    import os
    if not os.path.exists('./images/'):
        os.makedirs('./images/')

    imagefile.save(image_path)

    image = load_img("./images/" + imagefile.filename, target_size=(224, 224))

    image = img_to_array(image)
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    image = preprocess_input(image)
    yhat = model.predict(image)
    print(yhat)
    # label = decode_predictions(yhat)
    # label = label[0][0]

    # classification = '%s (%.2f%%)' % (label[1], label[2]*100)
    classification = '%f' % (yhat)

    return render_template('index.html', prediction=classification)


if __name__ == '__main__':
    app.run(debug=True, port=5002)
