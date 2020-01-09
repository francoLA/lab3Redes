import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

################################################# CONVOLUCION ######################################################

# convolucion para kernel de nxm e imagen de nxm
def convolucion(kernel, imagen):
    imagenFil = []
    for i in range(len(imagen) - len(kernel) + 1):
        aux = []
        for j in range(len(imagen[0]) - len(kernel[0]) + 1):
            aux.append(productoMatrices(kernel, imagen, i, j))
        imagenFil.append(aux)
    return extendMatrix(imagenFil)


#calculo de la suma de los valores de filas y columnas dadas
def productoMatrices(kernel, imagen, row, col):
    sum = 0
    for i in range(len(kernel)):
        for j in range(len(kernel[0])):
            sum = sum + kernel[i][j] * imagen[row + i][col + j]
    return sum


#agrega ceros al rededor de la matriz de entrada
def extendMatrix(matriz):
    nueva = []
    auxRow = []
    for i in range(len(matriz[0]) + 2):
        auxRow.append(0)
    nueva.append(auxRow)
    for row in matriz:
        auxRow = [0]
        for i in row:
            auxRow.append(i)
        auxRow.append(0)
        nueva.append(auxRow)
    auxRow = []
    for i in range(len(matriz[0]) + 2):
        auxRow.append(0)
    nueva.append(auxRow)
    return nueva


################################################## FILTRO SUAVIZADO GAUSSIANO ##################################################

#filtro gaussiano de la imagen
def filtroGaussiano(imagen):
    #se define el kernel del filtro
    kernel = ([[1.0, 4.0, 6.0, 4.0, 1.0],
               [4.0, 16.0, 24.0, 16.0, 4.0],
               [6.0, 24.0, 36.0, 24.0, 6.0],
               [4.0, 16.0, 24.0, 16.0, 4.0],
               [1.0, 4.0, 6.0, 4.0, 1.0]])

    kernel2 = []

    #se dividen los valores por 256
    for i in kernel:
        aux = []
        for j in i:
            j = j / 256
            aux.append(j)
        kernel2.append(aux)
    #se aplica la convolucion con el kernel y la imagen
    filtrada = convolucion(kernel2, imagen)

    generarGraficos('Filtro Gaussiano', 'Original', 'Filtro Gaussiano', imagen, filtrada)

    return filtrada


####################################################### FILTRO DETECTOR DE BORDES ##############################################

# filtro para bordes de la imagen
def filtroBordes(imagen):
    #se define el kernel del filtro
    kernel = [[1, 2, 0, -2, -1],
              [1, 2, 0, -2, -1],
              [1, 2, 0, -2, -1],
              [1, 2, 0, -2, -1],
              [1, 2, 0, -2, -1]]
    #se aplica la convolucion con el kernel y la imagen
    filtrada = convolucion(kernel, imagen)

    generarGraficos('Filtro de Bordes', 'Original', 'Filtro de Bordes', imagen, filtrada)

    return filtrada


# generar grafico
def generarGrafico(nombre, titulo, data):
    plt.figure(nombre)
    plt.title(titulo)
    plt.plot(data)


# calculo de la transformada de fourier de la imagen
def transformadaFourierImagen(imagen):
    f = np.fft.fft2(imagen)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum


# genera dos graficos para imagen 1 y 2
def generarGraficos(nombre, titulo1, titulo2, imagen1, imagen2):
    plt.figure(nombre)
    plt.subplot(121)
    plt.imshow(imagen1, cmap=plt.cm.gray)
    plt.title(titulo1)
    plt.subplot(122)
    plt.imshow(imagen2, cmap=plt.cm.gray)
    plt.title(titulo2)


#########################################BLOQUE PRINCIPAL O MAIN################################################################

imagen = imread('leena512.bmp')

gauss = filtroGaussiano(imagen)

bordes = filtroBordes(imagen)

transformadaFourierOriginal = transformadaFourierImagen(imagen)

transformmadaFourierGauss = transformadaFourierImagen(gauss)

transformmadaFourierBordes = transformadaFourierImagen(bordes)

generarGraficos('Dominio de la frecuencia imagen original', 'Original', 'Magnitud espectro', imagen, transformadaFourierOriginal)

generarGraficos('Dominio de la frecuencia filtro de bordes', 'Filtro de bordes', 'Magnitud espectro', bordes, transformmadaFourierBordes)


plt.show()
