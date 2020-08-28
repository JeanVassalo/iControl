import math
import numpy as np
import cv2
from .pupila import Pupila


class Olho(object):
    """
    This class creates a new imagem to isolate the eye and
    initiates the pupil detection.
    """

    PontosDoOlhoEsquerdo = [36, 37, 38, 39, 40, 41]
    PontosDoOlhoDireito = [42, 43, 44, 45, 46, 47]

    def __init__(self, imagemOriginal, pontos, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None

        self._analyze(imagemOriginal, pontos, side, calibration)

    @staticmethod
    def _middle_point(p1, p2):
        """Retorna o ponto médio (x, y) entre dois pontos

         Argumentos:
             p1 (dlib.point): Primeiro ponto
             p2 (dlib.point): Segundo ponto
        """
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def _isolate(self, frame, landmarks, points):
        """Isole um olho, para ter uma imagem sem outra parte do rosto.

         Argumentos:
             frame (numpy.ndarray): Frame contendo o rosto
             pontos de referência (dlib.full_object_detection): Pontos faciais para a região do rosto
             pontos (lista): Pontos de olho (a partir dos 68 pontos Multi-PIE)
        """
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)

        # Applying a mask to get only the eye
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # Cropping on the eye
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points):
        """Calcula uma proporção que pode indicar se um olho está fechado ou não.
         É a divisão da largura do olho, pela sua altura.

         Argumentos:
             pontos de referência (dlib.full_object_detection): Pontos faciais para a região do rosto
             pontos (lista): Pontos de olho (a partir dos 68 pontos Multi-PIE)

         Retorna:
             A proporção calculada
        """
        left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
        bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))

        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None

        return ratio

    def _analyze(self, original_frame, landmarks, side, calibration):
        """Detecta e isola o olho em uma nova imagem, envia dados para o calibracao
         e inicializa o objeto Pupila.

         Argumentos:
             original_frame (numpy.ndarray): Frame passado pelo usuário
             landmarks (dlib.full_object_detection): Pontos faciais para a região do rosto
             side: indica se é o olho esquerdo (0) ou o olho direito (1)
             calibração (calibracao.Calibracao): Gerencia o valor do limite de binarização
        """
        if side == 0:
            points = self.PontosDoOlhoEsquerdo
        elif side == 1:
            points = self.PontosDoOlhoDireito
        else:
            return

        self.blinking = self._blinking_ratio(landmarks, points)
        self._isolate(original_frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupila(self.frame, threshold)
