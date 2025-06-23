import numpy as np
import scipy


def mru(posicion_inicial,velocidad):
  def posicion(t):
    return posicion_inicial + velocidad * np.array(t)
  return posicion


class Codigo_pseudoaleatorio(object):
  """ Código binario pseudoaleatorio creado a partir de una semilla

  Atributos:

  semilla: Semilla válida aceptada por np.random.default_rng
  numero_bits: Número de bits en el código
  numero_muestras_bit: Número de muestras por chip de código (1: +1, 0: -1)
  bits_codigo: Arreglo con los bits del código
  muestras_codigo: Arreglo de muestras del código modulado en 2-pam(pulse amplitud modulation) (1: +1, 0: -1)
  autocorrelacion: Arreglo con valores de autocorrelación de las muestras
  del código para lags entre -(N-1) y N-1 inclusive

  """
  def __init__(self,semilla,numero_bits,numero_muestras_bit):

    self.numero_bits = numero_bits
    self.semilla = semilla
    self.numero_muestras_bit = numero_muestras_bit
    rng = np.random.default_rng(self.semilla)
    self.merito = 0
    for i in range(1000000):
      bits_codigo = rng.binomial(1,0.5,self.numero_bits)
      sx = np.array([-1.0,1.0])[bits_codigo]
      acf_abs = np.abs(np.correlate(sx,sx,'full'))
      #merito = 1/(np.sort(acf_abs)[-1])
      merito = 1/np.linalg.norm(acf_abs,ord=2)  #calcula el inverso de la norma 2 del vector de valor absoluto de la autocorrelación
      if merito > self.merito:
        self.bits_codigo = bits_codigo
        self.merito=merito
    sx = np.array([-1.0,1.0])[self.bits_codigo]
    self.muestras_codigo = np.kron(sx,[1]*numero_muestras_bit)
    #self.autocorrelacion = np.correlate(self.muestras_codigo,self.muestras_codigo,'full')

def interpola_sinc(secuencia,tiempo_muestreo):
    """Interpolación sinc $\dfrac{\sin(\pi x)}{\pi x}$"""
    secuencia = np.reshape(secuencia,(1,-1))
    n = np.reshape(np.arange(secuencia.size),(1,-1))
    def f(t):
        t = np.reshape(t,(-1,1))
        return np.reshape(np.sum(secuencia*np.sinc(t/tiempo_muestreo-n),axis=1),(-1,))
    return f

class Canal_radar(object):
  """ Código que recibiría un radar: señal + ruido

  Atributos:

  frec_muestreo: frecuencia con la que se muestrea la señal de tiempo discreto (por ejemplo 100 kHz = 100000 muestras por segundo) 
  pulso: muestras del código enviado
  guarda: distancia correspondiente al tiempo de guarda
  rango_maximo: máxima distancia esperada de llegada del radar
  rango_por_unidad_de_retardo: distancia recorrida por la señal por unidad de retardo, es decir distancia =delta t /2, donde delta t =1 por ser la unidad de retardo
  muestras_por_pulso: longitud del vector pulso = cantidad de chip que contiene el código enviado
  rango_minimo_sin_guarda: rango mínimo necesario para emitir el pulso completo, en metros
  rango_minimo: rango mínimo incluyendo tiempo de guarda
  amplitud_de_referencia_rango_minimo: Pico de correlación, fijado arbitrariamente como 1, para que en escala dB corresponda a 0.
  muestras_escucha: total de muestras obtenidas durante el tiempo de escucha (muestras_por_pulso + muestras_diferencia_retardo)
  vector_rx: arreglo de ceros de longitud muestras_escucha donde se simulará una señal recibida
  """
  def __init__(self,frec_muestreo,frec_portadora, pulso, guarda,rango_maximo,semilla_rng):
    self.frec_muestreo = frec_muestreo
    self.frec_portadora = frec_portadora   
    self.pulso = pulso
    self.pulso_interpolado = interpola_sinc(pulso,1/frec_muestreo)
    self.guarda = guarda
    self.rango_maximo = rango_maximo
    self.rng = np.random.default_rng(semilla_rng)

    periodo_de_muestreo = 1/frec_muestreo

    ## velocidad de la luz en m/s
    C = 3e8

    ## Distancia a un blanco hipotético tal que el eco arribe al transmisor con un retardo unitario
    ## sirve para convertir entre retardo y rango correspondiente de un eco
    self.rango_por_unidad_de_retardo = C/2

    self.muestras_por_pulso = pulso.size

    ## rango al blanco más cercano cuyo eco no se solapa con la señal transmitida
    self.rango_minimo_sin_guarda = self.muestras_por_pulso * periodo_de_muestreo * self.rango_por_unidad_de_retardo

    ## La guarda corresponde al tiempo entre el fin de la transmisión y el inicio de la recepción
    self.rango_minimo = self.rango_minimo_sin_guarda + self.guarda

    ## tiempo entre el inicio de la transmisión y el inicio de la recepción
    tiempo_tx_rx = self.rango_minimo/self.rango_por_unidad_de_retardo

    ## valor normalizado de la amplitud del pico de correlación del eco producido
    ## por un blanco de referencia ubicado en el rango mínimo
    self.amplitud_de_referencia_rango_minimo = 1

    ## retardo con que arriba el eco producido por un blanco en rango mínimo
    retardo_blanco_mas_cercano = self.rango_minimo/self.rango_por_unidad_de_retardo

    ## retardo con que arriba el eco producido por un blanco en rango máximo
    retardo_blanco_mas_lejano = rango_maximo/self.rango_por_unidad_de_retardo

    ## Duración de un pulso de radar
    tiempo_pulso = self.muestras_por_pulso * periodo_de_muestreo

    ## tiempo de escucha mínimo que permite recibir el eco de un blanco a rango
    ## máximo, si se inicia la recepción al terminar el tiempo de guarda.
    tiempo_minimo_de_escucha = retardo_blanco_mas_lejano - retardo_blanco_mas_cercano + tiempo_pulso

    ## Cantidad de muestras requeridas para cubrir el tiempo de escucha mínimo
    self.muestras_escucha = int(np.ceil(tiempo_minimo_de_escucha * frec_muestreo))
    self.vector_rx=np.zeros((self.muestras_escucha,),np.complex128)

  def nueva_exploracion(self):
    """
    Borra los datos generados en la ventana de escucha (vector_rx)

    Note
    ----
    Modifica vector_rx

    Returns
    -------
    objeto Radar sobre el que fue invocado (self)
    """
    self.vector_rx[:]=0
    return self
  def obt_tiempos_muestras_escucha(self):
    ## Tiempo correspondiente a la transmisión y el tiempo de guarda, durante el
    ## cual no hay recepción
    tiempo_no_registrado = self.rango_minimo / self.rango_por_unidad_de_retardo
    return np.arange(self.vector_rx.size)/self.frec_muestreo + tiempo_no_registrado

  def agrega_eco(self,rango_objetivo):
    """
    Registra un nuevo eco, sumándolo al vector de escucha (vector_rx)

    Note
    ----
    Modifica vector_rx

    Parameters
    ----------
    rango_objetivo: distancia a la que se encuentra un posible blanco

    Returns
    -------
    objeto Radar sobre el que fue invocado (self)

    """

    ## Aserción de que el rango de objetivo solicitado pertenece al rango observable
    assert rango_objetivo >= self.rango_minimo and rango_objetivo <= self.rango_maximo

    ## Amplitud que debe tener el eco recibido de un blanco ubicado en rango mínimo
    amplitud_rango_minimo = self.amplitud_de_referencia_rango_minimo / self.muestras_por_pulso

    ## Amplitud del eco producido por un blanco en el rango solicitado
    amplitud_rx = amplitud_rango_minimo * (self.rango_minimo/rango_objetivo)**2

    ## Retardo del eco debido a un blanco en el rango solicitado
    retardo_del_eco = rango_objetivo / self.rango_por_unidad_de_retardo

    t = self.obt_tiempos_muestras_escucha()
    #fase con la que llega la señal reflejada
    fase = np.exp(-1j*self.frec_portadora*2*np.pi*retardo_del_eco)
    #agrega a la señal original la señal atenuada
    self.vector_rx += self.pulso_interpolado(t-retardo_del_eco)*amplitud_rx*fase
    return self

  def agrega_ruido(self,snr_rango_maximo):
    """Función que agrega ruido gaussiano a la señal recibida de media 0 y desvío
    prefijado.

    Note
    -----
    Modifica `vector_rx`

    Parameters
    ---------
    snr_rango_maximo: Relación señal/ruido para rango máximo en veces respecto al eco del blanco estándar

    Returns
    -------
    objeto Radar sobre el que fue invocado (self)

    """
    ## Amplitud que debe tener el eco recibido de un blanco ubicado en rango mínimo
    amplitud_rango_minimo = self.amplitud_de_referencia_rango_minimo / self.muestras_por_pulso

    rango_objetivo = self.rango_maximo

    ## Amplitud del eco producido por un blanco en el rango solicitado
    amplitud_rx = amplitud_rango_minimo * (self.rango_minimo/rango_objetivo)**2

    # Variabilidad del ruido
    valor_eficaz_ruido = np.sqrt((amplitud_rx**2/2) / snr_rango_maximo) 
    
    valor_eficaz_por_componente = valor_eficaz_ruido / 2**.5
    ## A cada valor del array original vector_tx le suma un valor aleatorio de
    ## una Normal de media 0 y desvío estándar dado por el parámetro desv_est_ruido.
    self.vector_rx+=self.rng.normal(0,valor_eficaz_por_componente,self.vector_rx.size)+1j*self.rng.normal(0,valor_eficaz_por_componente,self.vector_rx.size)
    return self
  def copia_vector_rx(self):
    """
    Copia del vector de recepción actual
    """
    return self.vector_rx.copy()

class Compresor_pulso(object):
  """
  Calcula la correlación entre pulso emitido y señal recibida.

  """

  def __init__(self, canal_radar : Canal_radar):
    """
    canal_radar: Objeto Canal_radar de donde se toman los datos necesarios
                 para inicializar el compresor de pulso
    """
    self.pulso = canal_radar.pulso
    rango_muestra_km = (canal_radar.rango_por_unidad_de_retardo / canal_radar.frec_muestreo)/1000
    rango_minimo_km = canal_radar.rango_minimo/1000
    self.rango_minimo_km = rango_minimo_km
    self.frec_muestreo = canal_radar.frec_muestreo
  def correlacion(self,muestras_rx):
    """
    Correlaciona un vector de muestras recibidas con la forma del pulso
    transmitido
    """
    acf = np.correlate(muestras_rx,self.pulso,"full")
    return acf
  def amplitud_cuadrada_correlacion(self,muestras_rx):
    cor = np.abs(self.correlacion(muestras_rx))
    return np.abs(cor)**2
  
class Simulador_escenario(object):
  def __init__(self, canal: Canal_radar):
    self.canal = canal

  def exploracion(self,escenario, tiempo_inicial, periodo_de_repeticion_de_pulso, pulsos_por_exploracion,snr_rango_max=.5):
    """
    Simula la señal recibida a distintos tiempos dado un blanco en movimiento.
    escenario: lista de funciones de posición en función del tiempo para los distintos blancos.
    """
    resultados = []
    for t in [tiempo_inicial+k*periodo_de_repeticion_de_pulso for k in range(pulsos_por_exploracion)]:
      self.canal.nueva_exploracion()
      for posicion_blanco in escenario:
        posicion = posicion_blanco(t)
        self.canal.agrega_eco(posicion)
      self.canal.agrega_ruido(snr_rango_max)
      resultados.append(self.canal.copia_vector_rx())
    return np.array(resultados)
  
  def simula(self,escenario, tiempo_inicial, periodo_de_repeticion_de_pulso, pulsos_por_exploracion, numero_exploraciones,snr_rango_max=.5):
    """retorna lista de matrices doppler-rango para `numero_exploraciones` exploraciones"""
    return [np.fft.fft(self.exploracion(escenario, tiempo_inicial + periodo_de_repeticion_de_pulso*pulsos_por_exploracion*nx, periodo_de_repeticion_de_pulso, pulsos_por_exploracion,snr_rango_max),axis=0)
      for nx in range(numero_exploraciones)]
  