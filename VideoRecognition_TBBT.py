import cv2
import numpy as np
import tensorflow as tf
import os

# Set working directory
file_path = os.path.abspath("VideoRecognition_TBBT.py")
file_dir = os.path.dirname(file_path)
os.chdir(file_dir)

# Configuration
MODEL_PATH = 'emotion_model_weights.keras'
# Insert name of video file
VIDEO_SOURCE = 'Clip_SheldonSmile.mp4' 
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def run_video_analysis():
    # 1. Load the trained model
    try:
        print("Loading model")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Load the Face Detector (based on Haar cascades)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 3. Start Video Processing
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {VIDEO_SOURCE}")
        return

    print(f"✅ Processing {VIDEO_SOURCE}. Press 'q' to quit.")


    
    while True:
        ret, frame = cap.read()
        
        # if ret is false, video has ended
        if not ret:
            print("Video end")
            break


    # Convert to Grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # scale smaller portions of image
        faces = face_cascade.detectMultiScale(gray_frame,
                                                scaleFactor=1.1, 
                                                minNeighbors=12,  # this should be high otherwise detects a "face" on Sheldon t-shirt
                                                minSize=(60, 60))

        for (x, y, w, h) in faces:
            # Preprocessing
            roi_gray = gray_frame[y:y+h, x:x+w]
            
            # Histogram equalization (to better detect faces with shadows)
            #roi_gray = cv2.equalizeHist(roi_gray)
            
            try:
                roi_resized = cv2.resize(roi_gray, (48, 48))
            except Exception as e:
                continue 

            # DEBUG: shows what CNN is actually seeing
            cv2.imshow('Debug', roi_resized)
            # Normalization
            img_pixels = roi_resized.astype('float32')
            img_pixels = np.expand_dims(img_pixels, axis=0) 
            img_pixels = np.expand_dims(img_pixels, axis=-1)

            # Prediction
            predictions = model.predict(img_pixels, verbose=0)
            max_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            predicted_emotion = CLASS_NAMES[max_index]

            # Visualization (the rectangle with the class label)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{predicted_emotion} ({int(confidence*100)}%)"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                
        # Show the video recognition in action
        cv2.imshow('Video emotion detector', frame)
        # cv2.waitkey() slows down the video (1 is default)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_video_analysis()