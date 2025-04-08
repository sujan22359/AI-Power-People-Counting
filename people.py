import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
import cvzone

st.title("People Tracking System using YOLO")
st.sidebar.header("Upload Video")
uploaded_file = st.sidebar.file_uploader("people1.avi", type=["mp4", "avi", "mov"])

output_folder = "processed_videos"
os.makedirs(output_folder, exist_ok=True)

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    model = YOLO("yolov8n.pt")
    names = model.model.names
    
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = os.path.join(output_folder, "output.avi")
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (1020, 600))

    count = 0
    cy1, cy2, offset = 261, 286, 8
    inp, enter, exp, exitp = {}, [], {}, []

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 3 != 0:
            continue
        
        frame = cv2.resize(frame, (1020, 600))
        results = model.track(frame, persist=True, classes=0)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()
            
            for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                if cy1 - offset < cy < cy1 + offset:
                    inp[track_id] = (cx, cy)
                if track_id in inp and cy2 - offset < cy < cy2 + offset:
                    if track_id not in enter:
                        enter.append(track_id)
                
                if cy2 - offset < cy < cy2 + offset:
                    exp[track_id] = (cx, cy)
                if track_id in exp and cy1 - offset < cy < cy1 + offset:
                    if track_id not in exitp:
                        exitp.append(track_id)
        
        cv2.line(frame, (440, 286), (1018, 286), (0, 0, 255), 2)
        cv2.line(frame, (438, 261), (1018, 261), (255, 0, 255), 2)
        cvzone.putTextRect(frame, f'ENTERPERSON: {len(enter)}', (50, 60), 2, 2)
        cvzone.putTextRect(frame, f'EXITPERSON: {len(exitp)}', (50, 160), 2, 2)
        
        out.write(frame)
        stframe.image(frame, channels="BGR")

    cap.release()
    out.release()
    
    st.sidebar.success(f"Processed video saved: {output_path}")
    with open(output_path, "rb") as f:
        st.sidebar.download_button("Download Processed Video", f, file_name="output.avi")
