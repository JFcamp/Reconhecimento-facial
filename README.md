# Reconhecimento de Emoções em Tempo Real

## Este projeto demonstra como usar a biblioteca DeepFace para detectar e reconhecer emoções em tempo real a partir de uma webcam.

## Funcionalidades

- Detecta rostos em tempo real usando OpenCV.
- Usa DeepFace para analisar a emoção dominante em cada rosto detectado.
- Mostra a emoção detectada na imagem em tempo real.

## Pré-requisitos

- Python 3.7 ou superior
- Pacotes Python: `opencv-python`, `cvzone`, `deepface`

Codigo :
````bash
import cv2
from cvzone.FaceDetectionModule import FaceDetector
from deepface import DeepFace

video = cv2.VideoCapture(0)
detector = FaceDetector()

while True:
    ret, img = video.read()

    if not ret:
        break

    img, bboxes = detector.findFaces(img, draw=True)

    if bboxes:
        for bbox in bboxes:
            x, y, w, h = bbox['bbox']
            # Extraindo a região do rosto
            face_region = img[y:y+h, x:x+w]
            # Detectando emoções
            try:
                result = DeepFace.analyze(face_region, actions=[
                                          'emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                cv2.putText(img, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"Erro na análise de emoções: {e}")

    cv2.imshow('Resultado', img)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()
````


## Este projeto demonstra como usar a biblioteca OpenCV e o módulo de detecção de rostos do cvzone para realizar a detecção de rostos em tempo real a partir de uma webcam.]


## Funcionalidades

- Detecta rostos em tempo real usando OpenCV.
- Usa DeepFace para analisar a emoção dominante em cada rosto detectado.
- Mostra a emoção detectada na imagem em tempo real.

  
````bash

import cv2
from cvzone.FaceDetectionModule import FaceDetector

video = cv2.VideoCapture(0)
detector = FaceDetector()

while True:
    ret, img = video.read()

    if not ret:
        break

    img, bboxes = detector.findFaces(img, draw=True)

    cv2.imshow('Resultado', img)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()
````
## Este projeto demonstra como usar a biblioteca OpenCV e o módulo de detecção de rostos do cvzone para realizar a detecção de rostos em tempo real a partir de uma webcam.(porem com desfoque de rosto


## Funcionalidades

- Detecta rostos em tempo real usando OpenCV.
- Usa DeepFace para analisar a emoção dominante em cada rosto detectado.
- Mostra a emoção detectada na imagem em tempo real.
- Rosto desfocado
  
````bash
import cv2
from cvzone.FaceDetectionModule import FaceDetector

# Iniciando a captura de vídeo
video = cv2.VideoCapture(0)
detector = FaceDetector()

while True:
    # Capturando a imagem do vídeo
    ret, img = video.read()

    # Verificando se a imagem foi capturada corretamente
    if not ret:
        break

    # Fazendo uma cópia da imagem original para desfocar
    img_blur = img.copy()

    # Detectando faces na imagem
    img_blur, bboxes = detector.findFaces(img_blur, draw=False)

    # Desfocando os rostos detectados
    if bboxes:
        for bbox in bboxes:
            x, y, w, h = bbox['bbox']
            # Extraindo a região do rosto
            face_region = img_blur[y:y+h, x:x+w]
            # Aplicando desfoque gaussiano na região do rosto
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
            # Substituindo a região original pelo rosto desfocado
            img_blur[y:y+h, x:x+w] = blurred_face

    # Concatenando a imagem original e a imagem desfocada lado a lado
    result = cv2.hconcat([img, img_blur])

    # Mostrando o resultado
    cv2.imshow('Resultado', result)

    # Parando o loop quando a tecla 'ESC' (código 27) for pressionada
    if cv2.waitKey(1) == 27:
        break

# Liberando a captura de vídeo e fechando todas as janelas
video.release()
cv2.destroyAllWindows()
````
