import streamlit as st
from PIL import Image
import pytesseract
import pyttsx3
import torch  # PyTorch for YOLOv5
import cv2
import numpy as np
import requests
from io import BytesIO

# Title with Eye Icon
st.title("👁️ AI-Powered Assistive Solution for Visually Impaired 👁️")

# Upload Image
uploaded_image = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the uploaded image to a format suitable for detection (OpenCV format)
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Feature Selection
    feature = st.selectbox("Choose a feature", ["🌍 Real-Time Scene Understanding", "🔊 Text-to-Speech", "🔍 Object Detection", "🛠️ Personalized Assistance"])

    if feature == "🌍 Real-Time Scene Understanding":
        st.subheader("🌍 Scene Understanding")
        # Integrate a proper image captioning model or API here.
        caption = "This is a placeholder for scene captioning."
        st.write(f"Caption: {caption}")

    elif feature == "🔊 Text-to-Speech":
        st.subheader("🔊 Text-to-Speech Conversion")
        text = pytesseract.image_to_string(image)
        st.write(f"Extracted Text: {text}")
        tts = pyttsx3.init()
        tts.say(text)
        tts.runAndWait()

    elif feature == "🔍 Object Detection":
        st.subheader("🔍 Object Detection")
        # Load YOLOv5 pre-trained model for object detection (or use another model of choice)
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use small model for faster results
        results = model(opencv_image)

        # Process and display the detected objects
        image_with_boxes = results.render()[0]  # Draw bounding boxes on the image
        st.image(image_with_boxes, caption="Detected Objects", use_column_width=True)
        
        # Get labels of detected objects
        detected_objects = results.names
        # Convert tensor to numpy array and then to list for hashability
        detected_labels = [detected_objects[int(label)] for label in results.pred[0][:, -1].cpu().numpy()]
        st.write("Detected Objects:", detected_labels)

    elif feature == "🛠️ Personalized Assistance":
        st.subheader("🛠️ Personalized Assistance for Daily Tasks")

        # Load YOLOv5 model and detect objects
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load the small model
        results = model(opencv_image)
        
        # Get labels of detected objects
        detected_labels = [results.names[int(label)] for label in results.pred[0][:, -1].cpu().numpy()]
        
        # Personalized assistance based on detected objects with symbols
        assistance_text = ""
        for obj in detected_labels:
            if obj == "bottle":
                assistance_text += "🍶 This is a bottle. You might want to check if it's your favorite drink.\n"
            elif obj == "book":
                assistance_text += "📖 This is a book. Would you like me to read it aloud?\n"
            elif obj == "laptop":
                assistance_text += "💻 This is a laptop. Do you need help opening your files?\n"
            elif obj == "apple":
                assistance_text += "🍎 This is an apple. Would you like a reminder to wash it before eating?\n"
            elif obj == "phone":
                assistance_text += "📱 This is a phone. Would you like me to read your notifications?\n"
            elif obj == "pen":
                assistance_text += "🖊️ This is a pen. You might want to use it to jot something down.\n"
            elif obj == "cup":
                assistance_text += "☕ This is a cup. Do you need a reminder to refill your drink?\n"
            else:
                assistance_text += f"🔍 This is a {obj}. Let me know if you need help with it.\n"
        
        st.write(assistance_text)

        # Read out the personalized assistance text
        tts = pyttsx3.init()
        tts.say(assistance_text)
        tts.runAndWait()
