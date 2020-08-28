import cv2
from RastreioDoOlhar import InformacoesDaImagem

olhar = InformacoesDaImagem()
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Coletando cada uma das imagens que a webcam fornece
    _, imagem = webcam.read()

    # Enviando cada imagem para ser analisada
    olhar.proximaImagem(imagem)

    imagem = olhar.pupilasLocalizadas()
    texto = ""

    if olhar.piscando():
        texto = "Piscando"
    elif olhar.olhandoParaDireita():
        texto = "Olhando Para a Direita"
    elif olhar.olhandoParaEsquerda():
        texto = "Olhando Para a Esquerda"
    elif olhar.olhandoParaCentro():
        texto = "Olhando Para o Centro"

    cv2.putText(imagem, texto, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 255, 31), 2)

    pupilaEsquerda = olhar.coordenadaDaPupilaEsquerda()
    pupilaDireita = olhar.coordenadaDaPupilaDireita()
    cv2.putText(imagem, "Pupila Esquerda:  " + str(pupilaEsquerda), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 31), 1)
    cv2.putText(imagem, "Pupila Direita: " + str(pupilaDireita), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 31), 1)

    cv2.imshow("Demo", imagem)

    if cv2.waitKey(1) == 27:
        break
webcam.release()                         #libera a captura de tela
cv2.destroyAllWindows()                 #faz a liberação de memória