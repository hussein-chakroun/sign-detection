import sys
import time
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from gtts import gTTS
import os
import threading
from collections import deque


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SignLanguageTranslator:
    _instance = None
    _window_created = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SignLanguageTranslator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path=None):
        if self._initialized:
            return

        self._initialized = True
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot access webcam")

        self.sign_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        self.model = None
        if model_path:
            self.model = tf.keras.models.load_model(model_path)

        self.prediction_buffer = deque(maxlen=10)
        self.last_prediction = None
        self.prediction_threshold = 0.8

        self.speech_queue = deque()
        self.speech_thread = threading.Thread(target=self._process_speech_queue, daemon=True)
        self.speech_thread.start()

    def capture_screen(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def preprocess_landmarks(self, landmarks):
        landmark_array = np.array([(lm.x, lm.y, lm.z) for lm in landmarks]).flatten()
        return landmark_array.reshape(1, -1)

    def detect_and_predict(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        prediction = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                if self.model:
                    
                    h, w, c = image.shape
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0

                    for landmark in hand_landmarks.landmark:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)

                    
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)

                    
                    hand_img = image[y_min:y_max, x_min:x_max]

                    if hand_img.size > 0:
                        
                        hand_img = cv2.resize(hand_img, (64, 64))
                        hand_img = hand_img / 255.0  
                        hand_img = np.expand_dims(hand_img, axis=0)  

                        
                        raw_prediction = self.model.predict(hand_img, verbose=0)
                        predicted_class = self.sign_classes[np.argmax(raw_prediction)]
                        confidence = np.max(raw_prediction)

                        if confidence > self.prediction_threshold:
                            self.prediction_buffer.append(predicted_class)

                        if len(self.prediction_buffer) == self.prediction_buffer.maxlen:
                            current_prediction = max(set(self.prediction_buffer),
                                                    key=self.prediction_buffer.count)

                            if current_prediction != self.last_prediction:
                                self.speech_queue.append(current_prediction)
                                self.last_prediction = current_prediction
                            prediction = current_prediction

        return image, prediction

    def _process_speech_queue(self):
        while True:
            if self.speech_queue:
                text = self.speech_queue.popleft()
                self.speak_text(text)
            time.sleep(0.1)

    def speak_text(self, text):
        try:
            tts = gTTS(text=text, lang='en')
            tts.save("temp_speech.mp3")
            os.system("start temp_speech.mp3")
            time.sleep(1)
            try:
                os.remove("temp_speech.mp3")
            except:
                pass
        except Exception as e:
            print(f"Error in text-to-speech: {e}")

    def run(self):
        try:
            if not SignLanguageTranslator._window_created:
                cv2.namedWindow("Sign Language Translation", cv2.WINDOW_NORMAL)
                SignLanguageTranslator._window_created = True

            while True:
                frame = self.capture_screen()
                if frame is None:
                    continue

                processed_image, prediction = self.detect_and_predict(frame)

                if prediction:
                    cv2.putText(processed_image, f"Detected Sign: {prediction}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

                cv2.imshow("Sign Language Translation", processed_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"Error in run method: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        try:
            if hasattr(self, 'hands'):
                self.hands.close()
            if hasattr(self, 'cap'):
                self.cap.release()
            if SignLanguageTranslator._window_created:
                cv2.destroyWindow("Sign Language Translation")
                SignLanguageTranslator._window_created = False
            cv2.destroyAllWindows()
            if os.path.exists("temp_speech.mp3"):
                try:
                    os.remove("temp_speech.mp3")
                except:
                    pass
        except Exception as e:
            print(f"Error during cleanup: {e}")

    @classmethod
    def has_window(cls):
        """Check if window exists"""
        return cls._window_created
def create_and_train_model_from_images(dataset_path):
    """
    Creates and trains a CNN model for sign language recognition using images
    """
    
    img_height, img_width = 64, 64

    
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        validation_split=0.2
    )

    
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        classes=classes  
    )

    
    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        classes=classes  
    )
    

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(26, activation='softmax')  
    ])

    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    
    history = model.fit(
        train_generator,
        epochs=2,
        validation_data=validation_generator
    )

    
    model_path = "sign_language_cnn_model.h5"
    model.save(model_path)

    return model_path
            
            
def main():
    translator = None
    try:
        print("Sign Language Translation System")
        print("--------------------------------")

        while True:
            print("\nOptions:")
            print("1. Run with existing model")
            print("2. Train new model and run")
            print("3. Run without model (hand detection only)")
            print("4. Exit")

            choice = input("\nEnter your choice (1-4): ")

            
            if translator is not None:
                translator.cleanup()

            if choice == '1':
                try:
                    model_path = "sign_language_cnn_model.h5"
                    if not os.path.exists(model_path):
                        print(f"Error: Model file {model_path} not found.")
                        continue

                    print(f"Initializing translator with existing model: {model_path}")
                    translator = SignLanguageTranslator(model_path)
                    translator.run()
                    break
                except Exception as e:
                    print(f"Error loading existing model: {e}")
                    continue

            elif choice == '2':
                try:
                    dataset_path = input("Enter the path to your dataset folder: ")
                    if not os.path.exists(dataset_path):
                        print(f"Error: Dataset path {dataset_path} not found.")
                        continue

                    print("Training new model...")
                    model_path = create_and_train_model_from_images(dataset_path)
                    print(f"Model trained and saved to: {model_path}")

                    print("Initializing translator with new model...")
                    translator = SignLanguageTranslator(model_path)
                    translator.run()
                    break
                except Exception as e:
                    print(f"Error training model: {e}")
                    continue

            elif choice == '3':
                try:
                    print("Running hand detection only mode...")
                    translator = SignLanguageTranslator()
                    translator.run()
                    break
                except Exception as e:
                    print(f"Error in hand detection mode: {e}")
                    continue

            elif choice == '4':
                print("Exiting program...")
                if translator:
                    translator.cleanup()
                sys.exit(0)

            else:
                print("Invalid choice. Please try again.")

    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        if translator:
            translator.cleanup()
        cv2.destroyAllWindows()
        
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)