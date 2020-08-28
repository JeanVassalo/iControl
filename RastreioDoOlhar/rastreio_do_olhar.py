from __future__ import division
import os
import cv2
import dlib
from .olho import Olho
from .calibracao import Calibracao


class InformacoesDaImagem(object):
    """
     Essa classe acompanha o olhar do usuário.
     Ela fornece informações úteis como a posição dos olhos
     e pupilas e permite saber se os olhos estão abertos ou fechados
    """

    def __init__(self):
        self.imagem = None
        self.olhoEsquerdo = None
        self.olhoDireito = None
        self.calibracao = Calibracao()

        # detectorDeFace
        self.detectorDeFace = dlib.get_frontal_face_detector()

        # preditor para marcar os pontos na face
        cwd = os.path.abspath(os.path.dirname(__file__))
        enderecoDoModelo = os.path.abspath(os.path.join(cwd, "modelo_treinado/shape_predictor_68_face_landmarks.dat"))
        self.preditor = dlib.shape_predictor(enderecoDoModelo)

    @property
    def coordenadasDasPupilas(self):
        """Verifica se as pupilas foram localizadas"""
        try:
            int(self.olhoEsquerdo.pupil.x)
            int(self.olhoEsquerdo.pupil.y)
            int(self.olhoDireito.pupil.x)
            int(self.olhoDireito.pupil.y)
            return True
        except Exception:
            return False

    def analisar(self):
        """detecta a face e instancia o objeto olho"""
        imagemConvertida = cv2.cvtColor(self.imagem, cv2.COLOR_BGR2GRAY)
        faces = self.detectorDeFace(imagemConvertida)

        try:
            landmarks = self.preditor(imagemConvertida, faces[0])
            self.olhoEsquerdo = Olho(imagemConvertida, landmarks, 0, self.calibracao)
            self.olhoDireito = Olho(imagemConvertida, landmarks, 1, self.calibracao)

        except IndexError:
            self.olhoEsquerdo = None
            self.olhoDireito = None

    def proximaImagem(self, frame):
        """Atualiza a imagem e a analisa.

        Arguments:
            frame (numpy.ndarray): a imagem em analise
        """
        self.imagem = frame
        self.analisar()

    def coordenadaDaPupilaEsquerda(self):
        """retorna as coordenadas da pupila esquerda"""
        if self.coordenadasDasPupilas:
            x = self.olhoEsquerdo.origin[0] + self.olhoEsquerdo.pupil.x
            y = self.olhoEsquerdo.origin[1] + self.olhoEsquerdo.pupil.y
            return (x, y)

    def coordenadaDaPupilaDireita(self):
        """Retorna as coordenadas da pupila direita"""
        if self.coordenadasDasPupilas:
            x = self.olhoDireito.origin[0] + self.olhoDireito.pupil.x
            y = self.olhoDireito.origin[1] + self.olhoDireito.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Retorna um número entre 0,0 e 1,0 que indica a
         direção horizontal do olhar. A extrema direita é 0,0,
         o centro é 0,5 e a extrema esquerda é 1,0
        """
        if self.coordenadasDasPupilas:
            pupil_left = self.olhoEsquerdo.pupil.x / (self.olhoEsquerdo.center[0] * 2 - 10)
            pupil_right = self.olhoDireito.pupil.x / (self.olhoDireito.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Retorna um número entre 0,0 e 1,0 que indica o
         direção vertical do olhar. O topo extremo é 0,0,
         o centro é 0,5 e o fundo extremo é 1,0
        """
        if self.coordenadasDasPupilas:
            pupil_left = self.olhoEsquerdo.pupil.y / (self.olhoEsquerdo.center[1] * 2 - 10)
            pupil_right = self.olhoDireito.pupil.y / (self.olhoDireito.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def olhandoParaDireita(self):
        """Retorna verdadeiro se o usuário estiver olhando para a direita"""
        if self.coordenadasDasPupilas:
            return self.horizontal_ratio() <= 0.35

    def olhandoParaEsquerda(self):
        """Retorna verdadeiro se o usuário estiver olhando para a esquerda """
        if self.coordenadasDasPupilas:
            return self.horizontal_ratio() >= 0.65

    def olhandoParaCentro(self):
        """Retorna verdadeiro se o usuário estiver olhando para o centro"""
        if self.coordenadasDasPupilas:
            return self.olhandoParaDireita() is not True and self.olhandoParaEsquerda() is not True

    def piscando(self):
        """Retorna verdadeiro se o usuário estiver piscando """
        if self.coordenadasDasPupilas:
            blinking_ratio = (self.olhoEsquerdo.blinking + self.olhoDireito.blinking) / 2
            return blinking_ratio > 3.8

    def pupilasLocalizadas(self):
        """Retorna a imagem principal as pupilas destacadas"""
        frame = self.imagem.copy()

        if self.coordenadasDasPupilas:
            color = (0, 255, 0)
            x_left, y_left = self.coordenadaDaPupilaEsquerda()
            x_right, y_right = self.coordenadaDaPupilaDireita()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame
