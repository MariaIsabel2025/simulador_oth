import numpy as np
import scipy


def pfa_os(k,M,T):
  """
  Parámetros:
    k: estadístico de orden (entre 1 y M)
    M: cantidad de celdas de la ventana
    T: factor de escala para fijar el umbral
  Return: el valor de la probabilidad

  """
  comb = scipy.special.comb #asigno a un nombre el método comb de scipy que calcula un número combinatorio
  fact = scipy.special.factorial #asigno a un nombre el método factorial de scipy que calcula el factorial de un número
  return k*comb(M,k)*fact(k-1)*fact(T+M-k)/fact(T+M)  #fórmula de cálculo de la Prob de FA según fórmula 11.22 (pag 642) texto de Barkat

  ####################

def calcula_t_para_pfa(M,k=None,pfa=1e-6):  #por defecto la pfa = 1e-6
  if k is None:
    k=(M*6)//7
  return scipy.optimize.newton(lambda x:pfa_os(k,M,x)-pfa,1,maxiter=50000)

def detectarOsCfar(radar,correlacion,detector):
  """
  radar: diccionario de parámetros físicos del radar. Debe contener:
    frec_muestreo: frecuencia de muestreo en 1/T T: unidad de tiempo
    ancho_pulso: cantidad de muestras en el pulso codificado (adimensional)
    distancia_ciega: Rango del primer blanco cuyo eco es recibido completo, en L L: unidad de longitud
    rango_por_unidad_de_retardo: Incremento de rango por cada unidad de tiempo de retardo (mitad de la velocidad luz efectiva) en L/T
                                Opcional, si es omitido se usa 1.5e8 m/s (T=s L=m)

  correlacion: arreglo de amplitudes cuadradas de correlaciones de la señal recibida

  detector: diccionario de parámetros del detector
    T: factor de escala para fijar el umbral, proviene del calculo basado en PFA, en base a los otros parámetros
    ancho_ventana: Parámetro del CFAR, ancho de ventana deslizante para establacer el nivel de potencia de clutter (adimensional), no incluye la posición central
    k: orden del estadístico, debe ser menor que ancho_ventana-1, si está ausente usa ((ancho_ventana-1)*6)//7
    celdas_guarda: número de celdas de guarda (no se cuentan para el ancho_ventana, como tampoco la muestra central). Por defecto 0

  Retorna rangos,detalle
    rangos: lista con rangos de los picos detectados (en L)
    detalle: diccionario con
      posiciones lista con los indices de los picos dentro del vector de amplitudes de correlacion
      umbrales : lista de umbrales calculados para cada posición de ventana deslizante
      centros : lista de posiciones centrales de ventana correspondientes a umbrales
  """
  def obtener_ventana(posicion):
    """
    Atributos:
    ## offset_central: mitad de ancho de ventana
    ventana: array con las celdas a analizar
    izq:
    cent: celda de análisis
    der:
    entorno: array con las celdas del entorno de comparación (sin el centro y sin celdas de guarda)

    retorna: cent, entorno, posicion de la muestra cent en la correlacion
    """
    guarda = detector["celdas_guarda"]
    ancho_ventana = detector["ancho_ventana"]
    assert(correlacion.size >= posicion + ancho_ventana + 2*guarda + 1)

    offset_central = (ancho_ventana + guarda + 1)//2
    ventana = correlacion[posicion:posicion+ancho_ventana+2*guarda+1]
    izq  = ventana[:offset_central-guarda]
    cent = ventana[offset_central]
    der  = ventana[(offset_central+guarda+1):]
    entorno = np.concatenate((izq,der))
    return cent,entorno,posicion+offset_central

  if "rango_por_unidad_de_retardo" not in radar:
    radar["rango_por_unidad_de_retardo"] = 1.5e8
  if "k" not in detector:
    detector["k"]=((detector["ancho_ventana"]-1)*6)//7
  if "celdas_guarda" not in detector:
    detector["celdas_guarda"]=0
  rangos = []
  detalle = {
    "posiciones":[],
    "umbrales":[],
    "centros":[]}

  rango_muestra = radar["rango_por_unidad_de_retardo"]/radar["frec_muestreo"]

  for pos in range(correlacion.size-(detector["ancho_ventana"]+1+2*detector["celdas_guarda"])):
    centro,entorno,pos_centro = obtener_ventana(pos)
    estadistico = np.sort(entorno)[detector["k"]-1] # estadísitco 1 (min) corresponde a pos 0
    umbral = estadistico * detector["T"]
    detalle["umbrales"].append(umbral)
    detalle["centros"].append(pos_centro)
    if (centro > umbral): #comparación de cada celda a estudiar y el umbral
      detalle["posiciones"].append(pos_centro)  # si True, cambia 0 por 1 en la posición
      rangos.append((pos_centro-(radar["ancho_pulso"]-1))*rango_muestra+radar["distancia_ciega"])
  return rangos,detalle 