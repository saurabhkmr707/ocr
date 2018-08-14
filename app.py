import cv2
import math
import imutils
import numpy as np
import json
import sys

from tkinter import *
from tkinter import filedialog
root = Tk()
root.withdraw()
root.fileName = filedialog.askopenfilename( filetypes = (("JPEG images", "*.jpg"),("All Files","*.*")) )
image_fileName = root.fileName


img = cv2.imread(image_fileName)

img1 = img [:int(img.shape[0]/3),:int(img.shape[1]/2)]
img2 = img [:int(img.shape[0]/3),int(img.shape[1]/2):int(img.shape[1])-1]


img3 = img.copy()
template = cv2.imread(r'template.jpg')
w=template.shape[0]
h=template.shape[1]

left_box = cv2.matchTemplate(img1,template,cv2.TM_SQDIFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(left_box)
top_left1 = min_loc
bottom_right1 = (top_left1[0] + w, top_left1[1] + h)
print('Initial top left box' , top_left1,bottom_right1)


right_box = cv2.matchTemplate(img2,template,cv2.TM_SQDIFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(right_box)
top_left2 = min_loc
bottom_right2 = (top_left2[0] + w, top_left2[1] + h)
print('Initial top right box',top_left2,bottom_right2)

#To rotate the image
t1 = top_left2[1] - top_left1[1]
t2 = int(img.shape[1]/2) + top_left2[0] - top_left1[0]
angle = np.arctan2(t1,t2)
angle = math.degrees(angle)
print('Angle to be rotated',angle)
            
rotated = imutils.rotate(img, angle)
#cv2.imshow('rot',rotated)
#cv2.waitKey(0)

img = rotated.copy()
cv2.imwrite('rot.jpg',img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,175,255,cv2.THRESH_BINARY)

for j in range(thresh.shape[1]-int(img.shape[1]/2)-top_left2[0]):
    px = thresh[(top_left2[1]-60),(int(img.shape[1]/2)+top_left2[0]+j)]
#    print(px,(top_left2[1]-60),(int(img.shape[1]/2)+top_left2[0]+j))
    if px == 255:
        break;

print('Left side cropping factors', int(img.shape[1]/2)+top_left2[0]+j , j)
  
img3 = img[:int(img.shape[1]/3),:int(img.shape[0]/2)]
left_box = cv2.matchTemplate(img3,template,cv2.TM_SQDIFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(left_box)
top_left1 = min_loc
bottom_right1 = (top_left1[0] + w, top_left1[1] + h)
print('After rotation top left box' , top_left1,bottom_right1)

img4 = img[int(2*img.shape[0]/3):,:int(img.shape[0]/2)]
bottom_left_box = cv2.matchTemplate(img4,template,cv2.TM_SQDIFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(bottom_left_box)
top_left3 = min_loc
bottom_right3 = (top_left3[0] + w, top_left3[1] + h)
print('After rotation bottom most left box' , top_left3,bottom_right3)

bottom_crop = int(2*img.shape[0]/3) + bottom_right3[1] + 41  #41 is correction value
print('Bottom cropping factor',bottom_crop)

dim = (img.shape[1] , 2339)
cut_img = img[:bottom_crop,]
img = cv2.resize(cut_img, dim, interpolation = cv2.INTER_AREA)
cv2.imwrite('bot.jpg',img)

cut_img = img[(top_left1[1]-127):,: int(img.shape[1]/2)+top_left2[0]+j]
print('After croping size' , cut_img.shape)

if top_left1[0] <=20 :
    add = cv2.imread(r'../Downloads/template.jpg')
    ver = add[0:cut_img.shape[0],0:(20-top_left1[0])]
    to_resize = np.concatenate((ver,cut_img), axis=1)
else :
    to_resize = cut_img[ :cut_img.shape[0], (top_left1[0]-20):]
    
print('After concatenation if req',to_resize.shape)
#cv2.imshow('to_re',to_resize)
#cv2.waitKey(0)

dim = (1654,2339)
fin_img = cv2.resize(to_resize, dim, interpolation = cv2.INTER_AREA)


print('Final resized shape' , fin_img.shape)

cv2.imwrite('fin.jpg',fin_img)


# In[2]:


from keras.models import load_model
alpha_num=load_model(r'alphanum2.model')
alpha_num.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

alpha=load_model(r'alphabets22.model')
alpha.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

numbers=load_model(r'numbers2.model')
numbers.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

alphabets = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
            'P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']

s=""


# In[3]:





# In[4]:


import pandas as pd

df = pd.read_csv('content_box_coorniates.csv')

tags = df.columns.tolist()

list1 = ['Employee Code','OPTY ID','Pincode','Telephone No','DOB','Preferred Mobile Number']  #field contains numbers only
list2 = ['TITLE','NAME','Maiden Name','Mothers Maiden Name','CITY','STATE','Father/Spouse Name','LINE 1','LINE 2','LINE 3']  #field contains charcters only
list3 = ['Prefferd E-mail Address']  #field contains both

print('1.Number 2.Charcter 3.AlphaNumerical')



print('Number Fields',list1)
print('Alphabet Fields',list2)
print('Alphanumerical Fields',list3)


# In[5]:



tags = df.columns.tolist()

output = {}

for tag in tags:
    x = df[tag].tolist()
    y1 = int(x[0])
    y2 = int(x[1])
    x = x[2:]
    ind = x.index(max(x))
    x = x[:ind+1]
    x = [round(y) for y in x]
    z = 0
    
    
    for i in range(len(x)-1):
        character = fin_img[y1:y2,x[i]:x[i+1]]
        ret,character = cv2.threshold(character,230,255,cv2.THRESH_BINARY_INV)
        #  Resizing the charcacter to 28x28 to pass it the model
        dim = (28,28)
        letter = cv2.resize(character, dim, interpolation = cv2.INTER_AREA)
        letter = cv2.cvtColor(letter,cv2.COLOR_BGR2GRAY)    
        
        if cv2.countNonZero(letter) == 0 :
            if z!=1:
                s = s + " "
                z = 1
            else :
                break; 
        else:
            x1 = np.array(letter)
            x1 = x1/255
            x1 = x1.reshape(1,28,28,1)
            if list1.__contains__(tag):
                classes = numbers.predict_classes(x1)
                s = s + alphabets[26+classes[0]]
            elif list2.__contains__(tag):
                classes = alpha.predict_classes(x1)
                s = s + alphabets[classes[0]]
            else :
                classes = alpha_num.predict_classes(x1)
                s = s + alphabets[classes[0]]
            z = 0
    
    print(tag,':',s)
    output[tag] = s
    s=""


# In[13]:


output


# In[17]:


from flask import Flask, render_template,request
app = Flask(__name__)

@app.route('/')
def result():
   dict = output
   return render_template('result.html', result = dict)

@app.route('/display', methods=["GET", "POST"])
def display_result():
    with open('file.json', 'w') as f:
        json.dump(request.form.to_dict(flat = False), f)
    return render_template('display.html')
    


if __name__ == '__main__':
    import random, threading, webbrowser
    port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)

    threading.Timer(1.25, lambda: webbrowser.open(url) ).start()

    app.run(port=port, debug=False)
    #app.run(debug = True)




