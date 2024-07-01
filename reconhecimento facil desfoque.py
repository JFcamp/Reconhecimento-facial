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
