nul not found
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading

class DrowsinessDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Drowsiness Detection System")
        self.root.geometry("800x600")
        
        # Load the trained model
        self.model = load_model('models/drowsiness_model.h5')
        
        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        self.setup_gui()
        
    def setup_gui(self):
        # Title
        title = tk.Label(self.root, text="Drowsiness Detection System", font=("Arial", 20, "bold"))
        title.pack(pady=10)
        
        # Buttons frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Upload Image", command=self.upload_image, 
                 bg="blue", fg="white", padx=20, pady=10).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="Upload Video", command=self.upload_video, 
                 bg="green", fg="white", padx=20, pady=10).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="Use Webcam", command=self.use_webcam, 
                 bg="orange", fg="white", padx=20, pady=10).grid(row=0, column=2, padx=5)
        
        # Display frame
        self.display_frame = tk.Label(self.root, bg="gray", width=600, height=400)
        self.display_frame.pack(pady=20)
        
        # Results label
        self.result_label = tk.Label(self.root, text="", font=("Arial", 12))
        self.result_label.pack()
        
    def predict_drowsiness(self, face_img):
        # Preprocess for model
        face_resized = cv2.resize(face_img, (224, 224))
        face_normalized = face_resized / 255.0
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        # Predict
        prediction = self.model.predict(face_batch, verbose=0)[0][0]
        return prediction < 0.5  # True if drowsy
        
    def predict_age(self, face_img):
        # Simple age estimation (random for demo)
        # In real implementation, you'd use an age prediction model
        return np.random.randint(20, 60)
        
    def process_image(self, image_path):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        drowsy_count = 0
        drowsy_ages = []
        
        for (x, y, w, h) in faces:
            face_img = img_rgb[y:y+h, x:x+w]
            
            # Predict drowsiness
            is_drowsy = self.predict_drowsiness(face_img)
            
            if is_drowsy:
                # Draw red rectangle for drowsy
                cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 3)
                age = self.predict_age(face_img)
                drowsy_ages.append(age)
                drowsy_count += 1
                cv2.putText(img_rgb, f"DROWSY, Age: {age}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                # Draw green rectangle for awake
                cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(img_rgb, "AWAKE", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show popup if drowsy people detected
        if drowsy_count > 0:
            ages_str = ", ".join([str(age) for age in drowsy_ages])
            messagebox.showwarning("Drowsiness Alert!", 
                                 f"Found {drowsy_count} drowsy person(s)\nAges: {ages_str}")
        
        # Display results
        self.display_image(img_rgb)
        self.result_label.config(text=f"Total people: {len(faces)}, Drowsy: {drowsy_count}")
        
    def display_image(self, img):
        # Resize image to fit display
        height, width = img.shape[:2]
        max_height = 400
        if height > max_height:
            scale = max_height / height
            width = int(width * scale)
            height = max_height
            img = cv2.resize(img, (width, height))
            
        # Convert to PIL format and display
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.display_frame.config(image=img_tk)
        self.display_frame.image = img_tk
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.process_image(file_path)
            
    def upload_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if file_path:
            self.process_video(file_path)
            
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        def video_loop():
            ret, frame = cap.read()
            if ret:
                # Process frame similar to image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    face_img = frame_rgb[y:y+h, x:x+w]
                    is_drowsy = self.predict_drowsiness(face_img)
                    
                    if is_drowsy:
                        cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (255, 0, 0), 3)
                        cv2.putText(frame_rgb, "DROWSY", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    else:
                        cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 3)
                        cv2.putText(frame_rgb, "AWAKE", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                self.display_image(frame_rgb)
                self.root.after(30, video_loop)
            else:
                cap.release()
                
        video_loop()
        
    def use_webcam(self):
        self.process_video(0)  # 0 for default webcam

if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessDetectionGUI(root)
    root.mainloop()