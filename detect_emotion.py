#Ae alqoritmi telefondadi axtarma yalannan
import sys
import os
import io
import cv2 #-->esas kitabxana(computer vision)
import numpy as np #--> obrabotka izo, "predskazaniye"
from tensorflow.keras.models import load_model #--> dlya zaqruzki modeli h5(osibkaya fikir verme, konflikt sohbeti)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')#-->Az dili ucun UTF
#proverka argumentov(v nasem slucae 2 dlya izo i dlya modeli '.h5')
if len(sys.argv) < 3: #-->gozlediyi 2 argument
    print("Недостаточно аргументов") #--> yoxdusa bu
    sys.exit(1) 

image_path = sys.argv[1] #--> put k izo,ini
model_path = sys.argv[2] #--> put k modeli,ini
#--> obyazatelno proverka naliciye faylov v pamyati (izonu yaddan cixarma-argument sefi verir srazu(A. Postolit 112 sehife))
if not os.path.exists(image_path): #--> esli danniy fayl ne sus,
    print("Image not found") #--> eto
    sys.exit(1)

if not os.path.exists(model_path): #-->esli net modeli
    print("Model not found") #-->eto
    sys.exit(1)
    
#zaqruzka modeli(iz .h5)
try:
    model = load_model(model_path) #--> dlya udobstava 
except Exception as e:
    print("Model load error:", str(e))
    sys.exit(1)
    #--> massiv emociy
emotions = ["Qəzəbli", "İyrənmə", "Qorxu", "Xoşbəxt", "Kədərli", "Neytral", "Təcüblü"]

try:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) #--> convert sohbeti(Namiq muellim, seri)
    img = cv2.resize(img, (48, 48)) #-->(test papkasindadi yaddan cixarma) sekili 48x48 rahatciliq max lvl
    img = img.reshape(1, 48, 48, 1) / 255.0 #--> piksellerin normalizaciyasi, it seklindeki seh buna gore oldu 
#Insallah sonuncu fraqment
#1) zapuskat ele, sirf obrabotka ucun
    pred = model.predict(img) 
    
    emotion = emotions[np.argmax(pred)] #--> kuleklediyi emosiyanin max indexi
    print(emotion)
except Exception as e:
    print("Prediction error:", str(e))
    sys.exit(1)
