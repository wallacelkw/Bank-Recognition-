import numpy as np
import cv2 as cv
import h5py
import pickle
from tkinter import filedialog
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image

#--------------------Load the file --------------------------------------------
# we can load model_KNN, model_SVM, model_MLP
model_knn = pickle.load(open('model_KNN.p','rb'))
model_mlp = pickle.load(open('model_MLP.p','rb'))
model_svm = pickle.load(open('model_SVM.p','rb'))
path = 'Picture.h5'
file = h5py.File(path,'r+')
X_data = np.array(file["/dataset"]).astype("uint8")
y_data = np.array(file["/label"]).astype("uint8")
file.close()
y = np.ravel(y_data)


# -------------------LDA feature Extraction ------------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
lda.fit(X_data,y_data.ravel())

#---------------------------------- TKINTER---------------------------------------
def select_image():
    global mylabel_knn
    global mylabel_mlp
    global mylabel_svm

    img_path=filedialog.askopenfilename()
    img=cv.imread(img_path)

    img_resized = cv.resize(img, (400, 200))
    img_gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
    inputimg=img_gray.flatten().reshape(1,-1)

    inputimg = lda.transform(inputimg)

    probability_knn=model_knn.predict_proba(inputimg)
    probability_mlp = model_mlp.predict_proba(inputimg)
    probability_svm = model_svm.predict_proba(inputimg)
    Categories=['BSN','Maybank','Public Bank']

    #-------loop for the categories infomation--------------
    label_knn=[]
    label_mlp = []
    label_svm = []
    for ind, val in enumerate(Categories):
        information_knn = probability_knn[0][ind] * 100
        label_knn.append((f"{val} = %.2f" % round(information_knn, 2))+'%')

    for ind, val in enumerate(Categories):
        information_mlp = probability_mlp[0][ind] * 100
        label_mlp.append((f"{val} = %.2f" % round(information_mlp, 2))+'%')

    for ind, val in enumerate(Categories):
        information_svm = probability_svm[0][ind] * 100
        label_svm.append((f"{val} = %.2f" % round(information_svm, 2))+'%')

    print("KNN: ",label_knn)
    print("MLP: ", label_mlp)
    print("SVM: ", label_svm)
    print("\n")

    #--------display the predicted image result-------------
    final_test_knn = int(model_knn.predict(inputimg))
    final_test_mlp = int(model_mlp.predict(inputimg))
    final_test_svm = int(model_svm.predict(inputimg))

    test_catagories = 'The Predicted Image is: '
    cat_knn = Categories[final_test_knn - 1]
    cat_mlp = Categories[final_test_mlp - 1]
    cat_svm = Categories[final_test_svm - 1]

    cont_knn = test_catagories + cat_knn + '\n\n'
    cont_mlp = test_catagories + cat_mlp + '\n\n'
    cont_svm = test_catagories + cat_svm + '\n\n'


    #------------Tkinter Widget-----------------------------------

    #---------Read the inserted image-------------------------
    image = Image.open(img_path)
    image = image.resize((500,200), Image.ANTIALIAS)
    imgdisplay = ImageTk.PhotoImage(image)

    # Create a Label Widget to display the text or Image
    Label(root,image=imgdisplay).grid(row=4,column=0)

    Label(root, text="KNN Prediction", font=("Helvetica", 14,"bold")).grid(row=5, column=0)
    Label(root, text="MLP Prediction", font=("Helvetica", 14, "bold")).grid(row=8, column=0)
    Label(root, text="SVM Prediction", font=("Helvetica", 14, "bold")).grid(row=11, column=0)

    Label(root, text="\n".join(map(str, label_knn)), font=("Helvetica", 12)).grid(row=6, column=0,sticky="S")
    Label(root, text="\n".join(map(str, label_mlp)), font=("Helvetica", 12)).grid(row=9, column=0,sticky="S")
    Label(root, text="\n".join(map(str, label_svm)), font=("Helvetica", 12)).grid(row=12, column=0,sticky="S")

    mylabel_knn.destroy()
    mylabel_knn = Label(root,text=cont_knn,font=("Helvetica",12,"bold"))
    mylabel_knn.grid(row=7,column=0)

    mylabel_mlp.destroy()
    mylabel_mlp = Label(root, text=cont_mlp, font=("Helvetica", 12, "bold"))
    mylabel_mlp.grid(row=10, column=0)

    mylabel_svm.destroy()
    mylabel_svm = Label(root, text=cont_svm, font=("Helvetica", 12, "bold"))
    mylabel_svm.grid(row=13, column=0)

    root.mainloop()




root = tk.Tk()
root.title("Image Classification Group 6-Ready")
root.geometry('600x1000')
paths = 'logo/logo.JPG'
load =Image.open(paths)
render = ImageTk.PhotoImage(load)

main_logo = load.resize((150,150), Image.ANTIALIAS)
logodisplay = ImageTk.PhotoImage(main_logo)
# Setting icon of master window
root.iconphoto(False, render)

Label(root, image=logodisplay).grid(row=0, column=0,pady=5)

mylabel_svm=Label(root)
mylabel_mlp=Label(root)
mylabel_knn=Label(root)
Label(root, text="Bank Recognition System by Group 6-Ready",font=("Helvetica",20,"bold")).grid(row=1,sticky=N,rowspan=2)
Button(root, text="Select Image", height=2, width=18, command=select_image).grid(row=3,column=0)
root.mainloop()
