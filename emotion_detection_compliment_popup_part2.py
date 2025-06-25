import os
import cv2
import numpy as np
import time
from deepface import DeepFace
import tkinter as tk
from tkinter import messagebox
from threading import Thread

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class EmotionDetector:
    def __init__(self):
        self.capture = None
        self.is_running = False
        self.last_popup_time = 0
        self.popup_cooldown = 10  
        self.frame_skip = 10 
        self.frame_count = 0
        self.current_emotion = "neutral"
        self.emotion_history = []  
        self.history_length = 5
        self.face_location = None 
  
        self.emotion_colors = {
            "angry": (0, 0, 255),   
            "disgust": (0, 140, 255),
            "fear": (0, 69, 255),
            "happy": (0, 255, 0),   
            "sad": (255, 0, 0),     
            "surprise": (0, 255, 255),
            "neutral": (255, 255, 0)
        }
        

        self.emotion_labels = [
            {"name": "Happy", "color": self.emotion_colors["happy"]},
            {"name": "Sad", "color": self.emotion_colors["sad"]},
            {"name": "Angry", "color": self.emotion_colors["angry"]},
            {"name": "Fear", "color": self.emotion_colors["fear"]},
            {"name": "Surprise", "color": self.emotion_colors["surprise"]},
            {"name": "Disgust", "color": self.emotion_colors["disgust"]},
            {"name": "Neutral", "color": self.emotion_colors["neutral"]}
        ]
    
    def start(self):
        """Initialize and start the webcam capture"""
        try:
            self.capture = cv2.VideoCapture(0)
            
            if not self.capture.isOpened():
                print("Error: Could not access webcam")
                return False
            

            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            
            print("Webcam started successfully!")
            self.is_running = True
            return True
            
        except Exception as e:
            print(f"Error starting webcam: {e}")
            return False
    
    def show_compliment(self):
        """Show a positive affirmation popup in a separate thread"""
        if time.time() - self.last_popup_time < self.popup_cooldown:
            return
        
        self.last_popup_time = time.time()
        
        compliments = [
            "You're amazing!",
            "You are stronger than you think!",
            "You're doing great! Keep going!",
            "You are loved and appreciated!",
            "You're a wonderful person!",
            "This feeling will pass. Be gentle with yourself.",
            "Take a deep breath. You've got this!",
            "Your resilience is inspiring!",
            "Remember your strength in past challenges.",
            "Small steps forward are still progress! "
        ]
        
        
        if self.current_emotion == "sad":
            compliment = np.random.choice(compliments[5:8])
        elif self.current_emotion == "angry":
            compliment = np.random.choice(compliments[4:7])
        elif self.current_emotion == "fear":
            compliment = np.random.choice(compliments[6:10])
        else:
            compliment = np.random.choice(compliments[0:5])
            
        def show_popup():
            try:
                root = tk.Tk()
                root.withdraw()
               
                root.attributes('-topmost', True)
                messagebox.showinfo("💖 A Reminder for You!", compliment)
                root.destroy()
            except Exception as e:
                print(f"Error showing popup: {e}")
        
      
        popup_thread = Thread(target=show_popup)
        popup_thread.daemon = True
        popup_thread.start()
    
    def get_stable_emotion(self, new_emotion):
        """Use a rolling history to stabilize emotion detection"""
        self.emotion_history.append(new_emotion)
        if len(self.emotion_history) > self.history_length:
            self.emotion_history.pop(0)
            
       
        emotion_counts = {}
        for emotion in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
      
        return max(emotion_counts, key=emotion_counts.get)
    
    def draw_emotion_legend(self, frame):
        """Draw a legend explaining the emotion colors"""
        frame_height, frame_width = frame.shape[:2]
        legend_x = 20
        legend_y = frame_height - 180
        
        
        cv2.rectangle(frame, (legend_x - 10, legend_y - 10), 
                     (legend_x + 150, legend_y + 160), (50, 50, 50), -1)
        
        
        cv2.rectangle(frame, (legend_x - 10, legend_y - 10), 
                     (legend_x + 150, legend_y + 160), (255, 255, 255), 1)
        
        
        cv2.putText(frame, "Emotion Colors:", (legend_x, legend_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        
        for i, emotion in enumerate(self.emotion_labels):
            y_pos = legend_y + 35 + (i * 20)
            
            
            cv2.rectangle(frame, (legend_x, y_pos - 10), (legend_x + 15, y_pos + 5), 
                         emotion["color"], -1)
            
           
            cv2.putText(frame, emotion["name"], (legend_x + 25, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    def run(self):
        """Main processing loop"""
        if not self.start():
            return
        
        try:
           
            cv2.namedWindow("Emotion Detector", cv2.WINDOW_NORMAL)
            cv2.moveWindow("Emotion Detector", 100, 100)
            
            
            try:
                cv2.setWindowProperty("Emotion Detector", cv2.WND_PROP_TOPMOST, 1)
            except:
                pass  
            
            print("Press 'q' to quit")
            
            while self.is_running:
                isTrue, frame = self.capture.read()
                if not isTrue:
                    print("Failed to grab frame")
                    break
                    
                
                frame = cv2.flip(frame, 1)
                self.frame_count += 1
                
                
                if self.frame_count % self.frame_skip == 0:
                    try:
                        result = DeepFace.analyze(
                            frame, 
                            actions=['emotion'], 
                            detector_backend='opencv', 
                            enforce_detection=False
                        )
                        
                        
                        if isinstance(result, list):
                            if len(result) > 0:
                                result = result[0]
                            else:
                                continue
                        
                        new_emotion = result['dominant_emotion']
                        if 'region' in result and result['region']:
                            face_obj = result['region']
                            self.face_location = {
                                'x': face_obj['x'],
                                'y': face_obj['y'],
                                'w': face_obj['w'],
                                'h': face_obj['h']
                            }
                        
                     
                        self.current_emotion = self.get_stable_emotion(new_emotion)
                        
                       
                        if self.current_emotion in ["sad", "angry", "fear"] or np.random.random() < 0.05:
                            self.show_compliment()
                            
                    except Exception as e:
                        print(f"Error in emotion detection: {e}")
                        continue
                
               
                if self.face_location:
                    x = self.face_location['x']
                    y = self.face_location['y']
                    w = self.face_location['w']
                    h = self.face_location['h']
                 
                    color = self.emotion_colors.get(self.current_emotion, (0, 255, 0))
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                  
                    emotion_text = self.current_emotion.capitalize()
                    label_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    
                    cv2.rectangle(frame, (x, y-30), (x+label_size[0]+10, y), color, -1)
                    cv2.putText(frame, emotion_text, (x+5, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                
               
                self.draw_emotion_legend(frame)
                
               
                cv2.putText(frame, "Press 'q' to quit", 
                          (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
              
                cv2.putText(frame, f"Current: {self.current_emotion.capitalize()}", 
                          (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, self.emotion_colors.get(self.current_emotion, (255, 255, 255)), 
                          2, cv2.LINE_AA)
                
              
                cv2.imshow('Emotion Detector', frame)
                
              
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        self.is_running = False
        if self.capture:
            self.capture.release()
        cv2.destroyAllWindows()
        print("Emotion Detector stopped")


if __name__ == "__main__":
    try:
        detector = EmotionDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
