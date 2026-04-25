import cv2
import numpy as np
import tensorflow as tf

# --- Configuration ---
MODEL_PATH = 'emotion_model_weights.keras'
# These names must match the order in the dataset (0-6)
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def run_webcam():
    # 1. Load the trained model
    try:
        print("Loading model... (this might take a few seconds)")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Load the Face Detector (Haar Cascade)
    # This comes built-in with OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 3. Start Webcam
    # Try index 0 first (default camera). If it fails, try 1.
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("✅ Webcam started. Press 'q' to quit.")

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            break

        # Flip the frame horizontally (like a mirror) so it feels natural
        frame = cv2.flip(frame, 1)

        # Convert to Grayscale (Face detection and Model both need this)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        # scaleFactor=1.3 means "look for faces at different sizes"
        # minNeighbors=5 helps avoid false positives (detecting a lamp as a face)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # Loop through every face found
        for (x, y, w, h) in faces:
            # --- Preprocessing for the Model ---
            
            # 1. Crop the face region from the gray frame
            roi_gray = gray_frame[y:y+h, x:x+w]
            
            # 2. Resize to 48x48 (The strict requirement)
            try:
                roi_resized = cv2.resize(roi_gray, (48, 48))
            except Exception as e:
                continue # Skip if crop was invalid

            # 3. Add the necessary dimensions
            # Shape goes from (48, 48) -> (1, 48, 48, 1)
            img_pixels = np.expand_dims(roi_resized, axis=0) 
            img_pixels = np.expand_dims(img_pixels, axis=-1)

            # --- Prediction ---
            predictions = model.predict(img_pixels, verbose=0)
            max_index = np.argmax(predictions[0]) # Get index of highest score
            confidence = np.max(predictions[0])   # Get the confidence score
            predicted_emotion = CLASS_NAMES[max_index]

            # --- Visualization ---
            # Draw the box around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Write the emotion name and confidence above the face
            label = f"{predicted_emotion} ({int(confidence*100)}%)"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show the final image
        cv2.imshow('Emotion Detector (Press q to Quit)', frame)

        # Press 'q' on the keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()