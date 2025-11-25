import cv2
import tkinter as tk
from PIL import Image, ImageTk

# ------------------- GUI Setup -------------------
window = tk.Tk()
window.title("Sign Language Detection System")
window.geometry("900x700")

# Prediction Label
label_prediction = tk.Label(window, text="Prediction: ?", font=("Arial", 28), fg="blue")
label_prediction.pack(pady=20)

# Text Output Box (for forming words)
text_output = tk.Text(window, height=3, width=30, font=("Arial", 22))
text_output.pack(pady=20)

# ------------------- Webcam Setup -------------------
cap = cv2.VideoCapture(0)

label_video = tk.Label(window)
label_video.pack()

# Placeholder prediction function (will update later when model is ready)
def predict_gesture(landmarks=None):
    return "?"

# ------------------- Video Loop Function -------------------
def show_frame():
    ret, frame = cap.read()
    
    if not ret:
        print("Camera not detected!")
        return

    # Convert frame to RGB for display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to Tkinter display format
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    
    label_video.imgtk = imgtk
    label_video.configure(image=imgtk)

    # Get prediction and update GUI (placeholder now)
    predicted_letter = predict_gesture()
    label_prediction.config(text=f"Prediction: {predicted_letter}")

    window.after(10, show_frame)

# ------------------- Buttons -------------------
def add_to_text():
    current_prediction = label_prediction.cget("text").replace("Prediction: ", "")
    if current_prediction and current_prediction != "?":
        text_output.insert(tk.END, current_prediction)

def clear_text():
    text_output.delete("1.0", tk.END)

btn_add = tk.Button(window, text="Add Letter", font=("Arial", 18), command=add_to_text)
btn_add.pack(pady=10)

btn_clear = tk.Button(window, text="Clear Text", font=("Arial", 18), command=clear_text)
btn_clear.pack(pady=10)

# ------------------- Run Application -------------------
show_frame()
window.mainloop()
cap.release()
