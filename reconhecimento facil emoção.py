import cv2
from cvzone.FaceDetectionModule import FaceDetector
from deepface import DeepFace

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
    img, bboxes = detector.findFaces(img, draw=True)

    # Verificando se há rostos detectados
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
                # Exibindo a emoção na imagem
                cv2.putText(img, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"Erro na análise de emoções: {e}")

    # Mostrando o resultado
    cv2.imshow('Resultado', img)

    # Parando o loop quando a tecla 'ESC' (código 27) for pressionada
    if cv2.waitKey(1) == 27:
        break

# Liberando a captura de vídeo e fechando todas as janelas
video.release()
cv2.destroyAllWindows()
