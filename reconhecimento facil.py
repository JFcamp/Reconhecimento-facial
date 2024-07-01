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

    # Detectando faces na imagem
    img, bboxes = detector.findFaces(img, draw=False)

    # Desfocando os rostos detectados
    if bboxes:
        for bbox in bboxes:
            x, y, w, h = bbox['bbox']
            # Extraindo a região do rosto
            face_region = img[y:y+h, x:x+w]
            # Aplicando desfoque gaussiano na região do rosto
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
            # Substituindo a região original pelo rosto desfocado
            img[y:y+h, x:x+w] = blurred_face

    # Mostrando o resultado
    cv2.imshow('Resultado', img)

    # Parando o loop quando a tecla 'ESC' (código 27) for pressionada
    if cv2.waitKey(1) == 27:
        break

# Liberando a captura de vídeo e fechando todas as janelas
video.release()
cv2.destroyAllWindows()
