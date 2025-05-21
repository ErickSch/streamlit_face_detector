import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2

st.title("Live camera with face detection")

# Cargar el detector de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

captured_frame = st.session_state.get('captured_frame', None)

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None  # frame limpio para capturar
        self.processed_frame = None  # frame con recuadros para mostrar

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img.copy()  # Guardamos una copia limpia para captura

        # Ahora trabajamos en una copia para dibujar recuadros
        img_display = img.copy()

        h, w, _ = img_display.shape
        start_x = w // 4
        start_y = h // 4
        end_x = w * 3 // 4
        end_y = h * 3 // 4

        # Detección de rostro
        gray = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        face_inside_box = False

        for (x, y, fw, fh) in faces:
            # Verificar que TODA la cara esté dentro del marco
            if start_x < x and (x + fw) < end_x and start_y < y and (y + fh) < end_y:
                face_inside_box = True

            # # Dibujar recuadro azul en cada rostro detectado (opcional)
            # cv2.rectangle(img_display, (x, y), (x + fw, y + fh), (255, 0, 0), 2)

        # Dibujar recuadro guía de acuerdo al resultado
        color = (0, 255, 0) if face_inside_box else (0, 0, 255)
        cv2.rectangle(img_display, (start_x, start_y), (end_x, end_y), color, 3)

        self.processed_frame = img_display

        return av.VideoFrame.from_ndarray(img_display, format="bgr24")


ctx = webrtc_streamer(
    key="face-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_processor:
    if st.button("📸 Capturar foto"):
        captured_frame = ctx.video_processor.frame  # ahora frame limpio
        st.session_state['captured_frame'] = captured_frame

if captured_frame is not None:
    st.subheader("Imagen capturada:")
    st.image(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
