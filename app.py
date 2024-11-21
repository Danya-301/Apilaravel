from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import cv2

# Lista de nombres de aves
names = ['Amazona Alinaranja', 'Amazona de San Vicente', 'Amazona Mercenaria', 'Amazona Real', 
         'Aratinga de Pinceles', 'Aratinga de Wagler', 'Aratinga Ojiblanca', 'Aratinga Orejigualda', 
         'Aratinga Pertinaz', 'Batará Barrado', 'Batará Crestibarrado', 'Batara Crestinegro', 
         'Batará Mayor', 'Batará Pizarroso Occidental', 'Batará Unicolor', 'Cacatua Ninfa', 
         'Catita Frentirrufa', 'Cotorra Colinegra', 'Cotorra Pechiparda', 'Cotorrita Alipinta', 
         'Cotorrita de Anteojos', 'Guacamaya Roja', 'Guacamaya Verde', 'Guacamayo Aliverde', 
         'Guacamayo azuliamarillo', 'Guacamayo Severo', 'Hormiguerito Coicorita Norteño', 
         'Hormiguerito Coicorita Sureño', 'Hormiguerito Flanquialbo', 'Hormiguerito Leonado', 
         'Hormiguerito Plomizo', 'Hormiguero Azabache', 'Hormiguero Cantor', 'Hormiguero de Parker', 
         'Hormiguero Dorsicastaño', 'Hormiguero Guardarribera Oriental', 'Hormiguero Inmaculado', 
         'Hormiguero Sencillo', 'Hormiguero Ventriblanco', 'Lorito Amazonico', 'Lorito Cabecigualdo', 
         'Lorito de fuertes', 'Loro Alibronceado', 'Loro Cabeciazul', 'Loro Cachetes Amarillos', 
         'Loro Corona Azul', 'Loro Tumultuoso', 'Ojodefuego Occidental', 'Periquito Alas Amarillas', 
         'Periquito Australiano', 'Periquito Barrado', 'Tiluchí Colilargo', 'Tiluchí de Santander', 
         'Tiluchi Lomirrufo']

# Inicializar la aplicación Flask
app = Flask(__name__)

# Cargar el modelo
model = load_model('model_VGG16_v4.keras')

# Ruta para mostrar la interfaz
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para manejar la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Verificar que se ha subido un archivo de imagen
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    # Leer y procesar la imagen
    image_file = request.files['image']
    image = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))  # Ajustar tamaño según el modelo
    image = preprocess_input(np.expand_dims(image, axis=0))
    
    # Hacer la predicción
    preds = model.predict(image)
    predicted_class_index = np.argmax(preds)
    predicted_class_name = names[predicted_class_index]
    confidence_percentage = preds[0][predicted_class_index] * 100

    # Responder con la predicción
    return jsonify({
        "predicted_class": predicted_class_name,
        "confidence": confidence_percentage
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
