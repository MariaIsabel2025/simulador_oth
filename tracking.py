import numpy as np
# offsets of each variable in the state vector
iX = 0               #indexación de la posición en 0
iV = 1               #indexación de la velocidad en 1
NUMVARS = iV + 1     #número de variables en el sistema = 2


class KF(object):
    def __init__(self, initial_x: float,   #doy un valor inicial para la posición
                       initial_v: float,   #doy un valor inicial para la velocidad
                       accel_variance: float) -> None:
        # mean of state GRV
        self._x = np.zeros(NUMVARS)

        self._x[iX] = initial_x
        self._x[iV] = initial_v

        self._accel_variance = accel_variance

        # covariance of state GRV
        self._P = np.eye(NUMVARS)    #crea matriz identidad 2x2

    def predict(self, dt: float) -> None:
        # x = F x
        # P = F P Ft + G Gt a
        F = np.eye(NUMVARS)
        F[iX, iV] = dt
        new_x = F.dot(self._x)

        G = np.zeros((2, 1))
        G[iX] = 0.5 * dt**2
        G[iV] = dt
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._accel_variance

        self._P = new_P
        self._x = new_x

    def update(self, meas_value: float, meas_variance: float):
        # y = z - H x
        # S = H P Ht + R
        # K = P Ht S^-1
        # x = x + K y
        # P = (I - K H) * P

        H = np.zeros((1, NUMVARS))
        H[0, iX] = 1

        z = np.array([meas_value])
        R = np.array([meas_variance])

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self._x + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self._P)

        self._P = new_P
        self._x = new_x

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def mean(self) -> np.array:
        return self._x

    @property
    def pos(self) -> float:
        return self._x[iX]

    @property
    def vel(self) -> float:
        return self._x[iV]

class GestorDeTrayectos(object):
  def __init__(self):
    self._trayectos = []
    self._tiempo = 0
  def actualizaTrayectos(self,detecciones,dt,varDet,varAcel,tol=1,tolPred=10):
    """ detecciones: lista de rangos de detecciones
        dt : paso de tiempo
        varDet : varianza del detector en su estimación de rango
        varAcel : varianza de la aceleración del blanco (para nuevos trayectos)
        tol : tolerancia en desviaciones estandar
        tolPred : Si la predicción tiene un desvío estandar mayor que
                  (varDet**.5 * tolPred) el trayecto deja de considerarse activo
    """
    def estimadoPosVel(estimador:KF):
      posEstMedia =  estimador.pos
      posEstDesvio = estimador.cov[iX,iX]**.5
      velEstMedia =  estimador.vel
      velEstDesvio = estimador.cov[iV,iV]**.5
      return [self._tiempo,posEstMedia,posEstDesvio,velEstMedia,velEstDesvio]

    self._tiempo = self._tiempo + dt
    usado = np.array([False]*len(detecciones))
    indices = np.arange(usado.size)
    detecciones = np.array(detecciones)
    desvDet = varDet**.5
    for estimador,puntos,activo in self._trayectos:
      if not activo[0]: continue
      estimador.predict(dt)
      xPredMedia = estimador.pos
      xPredDesvio = estimador.cov[iX,iX]**.5
      if xPredDesvio > tolPred * desvDet:
        activo[0] = False
        continue
      xmin = xPredMedia - tol*(xPredDesvio+desvDet)
      xmax = xPredMedia + tol*(xPredDesvio+desvDet)
      seleccion = np.logical_and(np.logical_and(detecciones > xmin, detecciones < xmax),np.logical_not(usado))
      detViables = detecciones[seleccion]
      if detViables.size == 0: continue
      imin = np.argmin(np.abs(detViables-xPredMedia))
      estimador.update(detViables[imin],varDet)
      usado[indices[seleccion][imin]]=True
      puntos.append(estimadoPosVel(estimador))
    for detLibre in detecciones[np.logical_not(usado)]:
      estimador = KF(detLibre,0,varAcel)
      self._trayectos.append((estimador,[estimadoPosVel(estimador)],[True]))
  def copiaTrayectos(self):
    return [np.array(puntos) for _,puntos,activo in self._trayectos if activo[0]]
