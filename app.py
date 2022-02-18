
import tensorflow as tf
import io
import numpy as np
import base64
from flask import Flask , jsonify , request
from PIL import Image

model = tf.keras.models.load_model( 'models/model_age.h5' )
model.summary()

app = Flask( __name__ )

@app.route( "/predict" , methods=[ 'POST' ] )
def predict():
    image_base64 = request.get_json()[ 'image' ]
    image = Image.open( io.BytesIO( base64.b64decode( image_base64 ) ) ).resize( ( 200 , 200 ) ).convert( 'RGB')
    image = np.asarray( image )
    image = np.expand_dims( image , axis=0 ) / 255.0
    predictions = model.predict( image )
    return jsonify( prob=predictions[ 0 ][ 0 ] )

if __name__ == "__main__":
    app.run( debug=True )