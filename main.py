import os
from google.cloud import storage
import tensorflow as tf
from io import BytesIO
from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
from tensorflow.keras.applications.mobilenet import preprocess_input

app = Flask(__name__)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'fishify-credentials.json'
storage_client = storage.Client()


def req(y_true, y_pred):
    req = tf.metrics.req(y_true, y_pred)[1]
    tf.keras.backend.get_session().run(tf.local_variables_initializer())
    return req


# model_filename = 'my_model_fix.h5'
# model_bucket = storage_client.get_bucket('sa-lindungi-model-bucket')
# model_blob = model_bucket.blob(model_filename)
# model_blob.download_to_filename(model_filename)
# model = load_model(model_filename, custom_objects={'req': req})
model = load_model('image_classification_model.h5')

# Mapping of fish indices to fish names
fish_names = {
    0: 'Belut',
    1: 'Ikan Bawal',
    2: 'Ikan Gurame',
    3: 'Ikan Kembung',
    4: 'Ikan Lele',
    5: 'Ikan Mas',
    6: 'Ikan Nila',
    7: 'Ikan Patin',
    8: 'Ikan Tenggiri',
    9: 'Udang'
    # Add more fish names as needed
}

@app.route('/', methods=['POST'])
def predict_fish():
    if request.method == 'POST':
        try:
            image_bucket = storage_client.get_bucket('fishify')  # Change the bucket name
            filename = request.json['filename']
            img_blob = image_bucket.blob('upload_picture/' + filename)
            img_path = BytesIO(img_blob.download_as_bytes())
        except Exception as e:
            respond = jsonify({'message': f'Error loading image file: {str(e)}'})
            respond.status_code = 400
            return respond

        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        x = tf.keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        images = np.vstack([x])

        # model predict
        pred_fish_index = model.predict(images).argmax()
        predicted_fish_name = fish_names.get(pred_fish_index, 'Unknown Fish')

        result = {
            "fish_name": predicted_fish_name
        }

        respond = jsonify(result)
        respond.status_code = 200
        return respond

    return 'OK'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
