import torch
import cv2
import numpy as np

# ====== Model Setup ======
model_path = r"C:\Users\moham\Desktop\AI NIGHT CHALLENGE\PPE\best.pt"
model = torch.load(model_path)
model.eval()  # evaluation mode

# Classes
class_list = ['vest', 'helmet', 'worker']

# ====== Camera Setup ======
cap = cv2.VideoCapture(0)  # 0 = default webcam

# ====== Preprocessing Function ======
def preprocess(frame):
    # Resize frame to model input size (replace with your model's input size)
    frame_resized = cv2.resize(frame, (224, 224))
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    # Normalize to 0-1
    frame_rgb = frame_rgb / 255.0
    # Convert to torch tensor and add batch dimension
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float()
    return tensor

# ====== Real-time Prediction Loop ======
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    input_tensor = preprocess(frame)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        pred_class = class_list[pred.item()]  # map to human-readable class

    # Display prediction on frame
    cv2.putText(frame, f'Prediction: {pred_class}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow('PPE Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()