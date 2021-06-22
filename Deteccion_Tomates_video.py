#Importamos librerias
import cv2 as cv
import numpy as np
 
#funcion para dibujar los objetos encontrados
def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
 
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
 
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
     
        cv.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 0), 2)
        cv.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
 
##########################################################################################################################
#************************************************************************************************************************
#                                        CODIGO PRINCIPAL
whT = 320 # ancho y alto de la imagen
confThreshold =0.5 # umbral de precision o conincidencia
nmsThreshold= 0.2  # Umbral suprecion no maxima
 

#Cargamos las clases o etiquetas del detector
classesFile = "config_file/tomate.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


## Configuramos nuestra Red
modelConfiguration = "config_file/tomate.cfg"     #Archivo de arquitectura de la Red
modelWeights = "config_file/tomate.weights" # Pesos de la Red
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights) # Configuracion de la red
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV) # configuramos opencv por detras
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)      # configuramos que trabaje con cpu

#Lectura del video
cap = cv.VideoCapture("video/*.*")


while True:
    
    ret,frame = cap.read()
    
    if ret:
 
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        findObjects(outputs,img)

        cv.imshow('Image', img)
        
        if cv.waitKey(10) & 0xFF==ord("q"):
            break
        
    else:
        break
        
cap.release()        
cv.destroyAllWindows()
    
