from flask import Flask, request, render_template
from PIL import Image, ImageDraw, ImageFont
import face_recognition
import joblib
import os
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
UPLOAD_FOLDER = 'uploads'
DETECTED_FOLDER = 'detected_faces'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTED_FOLDER'] = DETECTED_FOLDER

# Load model
trained_model_path = "face_detection_model.joblib"
trained_encoder_path = "face_encoder_labels.joblib"

face_encodings = joblib.load(trained_model_path)
labels = joblib.load(trained_encoder_path)

# ML
face_model = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability=True))
face_model.fit(face_encodings, labels)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_faces(image_stream):
    # Use image_stream instead of saving to disk
    test_image = face_recognition.load_image_file(image_stream)
    face_locations = face_recognition.face_locations(test_image)

    if len(face_locations) > 0:
        pil_image = Image.fromarray(test_image)
        draw = ImageDraw.Draw(pil_image)

        detected_faces = []
        for i, face_location in enumerate(face_locations):
            top, right, bottom, left = face_location
            draw.rectangle([left, top, right, bottom], outline="red", width=2)

            face_encoding = face_recognition.face_encodings(test_image, [face_location])[0]
            prediction = face_model.predict_proba([face_encoding])
            label = face_model.classes_[prediction.argmax()]

            label_position = (left + (right - left) // 2, top - 20)
            font_size = 18 

            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

            draw.text(label_position, label, fill="red", font=font)
            face_image = pil_image.crop((left, top, right, bottom))
            face_image_path = os.path.join(app.config['DETECTED_FOLDER'], f"face_{i}_{label}.png")
            face_image.save(face_image_path)

            with open(face_image_path, "rb") as face_image_file:
                encoded_face_image = base64.b64encode(face_image_file.read()).decode("utf-8")

            detected_faces.append({
                "label": label,
                "face_image_path": face_image_path,
                "encoded_face_image": encoded_face_image
            })

        detected_image_path = os.path.join(app.config['DETECTED_FOLDER'], "detected_image.png")
        pil_image.save(detected_image_path)

        with open(detected_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        return detected_faces, detected_image_path, encoded_image
    else:
        return None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('result.html', error="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('result.html', error="No selected file")

    if file and allowed_file(file.filename):
        # Process the file using streaming
        detected_faces, detected_image_path, encoded_image = detect_faces(file.stream)

        if detected_faces is not None:
            return render_template('result.html', success="File uploaded and faces detected successfully",
                                   detected_faces=detected_faces, detected_image_path=detected_image_path,
                                   encoded_image=encoded_image)
        else:
            return render_template('result.html', error="No faces detected in the uploaded image")

    return render_template('result.html', error="Invalid file format")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
