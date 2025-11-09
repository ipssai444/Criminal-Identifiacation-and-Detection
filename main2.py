
import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import face_recognition
from datetime import datetime

def load_image(image_file):
    img = Image.open(image_file)
    return img
    

st.title("Criminal Identification and Detection Application")

img4 = "190506-face-recognition-small-crimes-main-kh.jpg"

st.image(img4)
st.subheader('Menu: Select what you want to do')
page = st.selectbox(' ',('Criminal Registry','Detect Criminals with Images','Video_Survellance','Detect Criminals on Video'))

#Upload Criminal Images for video real time identification
if page == "Criminal Registry":
    st.header("Criminal Registry")
    st.subheader("Upload Images of criminals for Video Surveillance Identification")        #criminals are registered here for video surveillance
    image_file = st.file_uploader("Upload an Image",type=['png','jpeg','jpg']) #types of images to upload
    if image_file is not None:
        file_details = {"FileName":image_file.name,"FileType":image_file.type} #filetypes and name of uploaded
        st.write(file_details)
        img = load_image(image_file)
        st.image(img)                  #display
        with open(os.path.join("Train_Images",image_file.name),"wb") as f:   # for saving uploaded image
            f.write(image_file.getbuffer())         
        st.success("Saved File")
    
        pathfinderImages = "Train_Images"            #show all files in Train_Images folder
        dirImage_list = os.listdir(pathfinderImages)
 
        st.write("Total Images")
        # prints all Images
        st.write(dirImage_list)
        
        
if page == "Detect Criminals with Images":
    st.header("Detect Criminals with Images")
    # Detect Criminals
    st.subheader("Upload the Prexisting image of the Criminal to be Detected")     
    image_file2 = st.file_uploader("Upload Prexisting Image.",type=['png','jpeg','jpg'])
    if image_file2 is not None:
        file_details2 = {"FileName":image_file2.name}
        st.write(file_details2)
        img2 = load_image(image_file2)
        st.image(img2)
        with open(os.path.join("Train_Images2",image_file2.name),"wb") as f: 
            f.write(image_file2.getbuffer())         
        st.success("Saved File")

    
    st.subheader("Upload the Current image of the Criminal to be Identified")
    image_file3 = st.file_uploader("Upload the Current Image to be detected",type=['png','jpeg','jpg'])
    if image_file3 is not None:
        file_details3 = {"FileName":image_file3.name}
        st.write(file_details3)
        img3 = load_image(image_file3)
        st.image(img3)
        with open(os.path.join("Test_Images",image_file3.name),"wb") as f: 
            f.write(image_file3.getbuffer())         
        st.success("Saved File")
   
    if(st.button("Detect Criminal Via Image")):
        pathfindTrain = 'C:/Users/panta/OneDrive/Documents/Face/Streamlit - Criminal/Train_Images2/Hafiz_Saeed.jpg'
        pathfindTest = 'C:/Users/panta/OneDrive/Documents/Face/Streamlit - Criminal/Test_Images/Hafeez Sayeed 2.jpg'
        img = face_recognition.load_image_file(pathfindTrain)    # load the image as train file for what the code should recognize
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)        # encoding
        imgTest = face_recognition.load_image_file(pathfindTest) # load test image
    
        faceLoc = face_recognition.face_locations(img)[0]   #load image 
        
        encode = face_recognition.face_encodings(img)[0]  #recognize and mark the image for box highlight over it.
        
        cv2.rectangle(img,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2) #highlight
 
        faceLocTest = face_recognition.face_locations(imgTest)[0]
        
        encodeTest = face_recognition.face_encodings(imgTest)[0]
        
        cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
 
        results = face_recognition.compare_faces([encode],encodeTest) #compare train and test image
        
        faceDis = face_recognition.face_distance([encode],encodeTest)
        print(results,faceDis)
        cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2) #for text on image
 
        st.image(imgTest)
        cv2.waitKey(0)
    
if page == "Video_Survellance":
    st.write("Put the Criminal infront of the web cam")
    if (st.button('Video_Survellance')):
        pathfind = 'Train_images'     #the path of the directory where training images are stored in folder
        allimages = []                 #image list
        Names = []         #number of images name
        myList = os.listdir(pathfind)
        print(myList)
        for clsr in myList:
            currentImg = cv2.imread(f'{pathfind}/{clsr}')            # store the images in the list
            allimages.append(currentImg)                          # after reading from directory
            Names.append(os.path.splitext(clsr)[0])
        print(Names)            #List of images without extension

        def Attendance(naming):                       #mark attendace in csv files
            with open('Criminal_record.csv', 'r+') as f:
                lineList = f.readlines()
                # read and append in the list images
                print(lineList)
                nameList = []
                for line in lineList:
                    entry = line.split(',')
                    nameList.append(entry[0])
                    if naming not in nameList:
                        now = datetime.now()
                        dataString = now.strftime('%H:%M:%S') # mark date and time
                        f.writelines(f'\n{naming},{dataString}') # write

        def Encodings(images):
            enList = []
            # convert to RGB format for recognzie and encode by python.

            for jmg in allimages:
                jmg = cv2.cvtColor(jmg, cv2.COLOR_BGR2RGB)              
                encode = face_recognition.face_encodings(jmg)[0]
                enList.append(encode)
                return enList


        enListKnowns = Encodings(allimages)   # marked and id the image in webcam
        print('Encoding Complete')

        cap = cv2.VideoCapture(0)
        # initialize web cam and check
        while True:
            success, jmg = cap.read()

            imgs = cv2.resize(jmg, (0, 0), None, 0.25, 0.25)
            imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

            facesCurrentFrame = face_recognition.face_locations(imgs)
            encodesCurrentFrame = face_recognition.face_encodings(imgs, facesCurrentFrame)

            for enFace, faceLocation in zip(encodesCurrentFrame, facesCurrentFrame):  #check each image frame
                matching = face_recognition.compare_faces(enListKnowns, enFace)
                faceDistance = face_recognition.face_distance(enListKnowns, enFace)

                matchingIndex = np.argmin(faceDistance)

                if matching[matchingIndex]:  # marking the box over image if matched
                    naming = Names[matchingIndex].upper()

                    y1, x2, y2, x1 = faceLocation
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(jmg, (x1, y1), (x2, y2), (255, 0, 0), 2)  #image rectangle
                    cv2.rectangle(jmg, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED) #text rectangle
                    cv2.putText(jmg, naming, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    Attendance(naming)

            cv2.imshow('Detector', jmg)
    
    
            if cv2.waitKey(1) == ord('q'):   # break loop by pressing q on keyboard
                break                
    
    
        cap.release()
        cv2.destroyAllWindows()    #Close the webcam window
    

if page == "Detect Criminals on Video":
    st.subheader("Upload the Video of the Criminal for Detection")
    image_file5 = st.file_uploader("Upload the Video",type=['mp4'])
    with open(os.path.join("TestVideo",image_file5.name),"wb") as f: 
        f.write(image_file5.getbuffer())         
    st.success("Saved File")
    
    st.write("Video Detection")
    if (st.button('Detect Criminals on Video')):
        pathfind = 'Train_images'     #the path of the directory where training images are stored in folder
        allimages = []                 #image list
        Names = []         #number of images name
        myList = os.listdir(pathfind)
        print(myList)
        for clsr in myList:
            currentImg = cv2.imread(f'{pathfind}/{clsr}')            # store the images in the list
            allimages.append(currentImg)                          # after reading from directory
            Names.append(os.path.splitext(clsr)[0])

        def Attendance(naming):                       #mark attendace in csv files
            with open('Criminal_record.csv', 'r+') as f:
                lineList = f.readlines()
                # read and append in the list images
                print(lineList)
                nameList = []
                for line in lineList:
                    entry = line.split(',')
                    nameList.append(entry[0])
                    if naming not in nameList:
                        now = datetime.now()
                        dataString = now.strftime('%H:%M:%S') # mark date and time
                        f.writelines(f'\n{naming},{dataString}') # write

        def Encodings(images):
            enList = []
            # convert to RGB format for recognzie and encode by python.

            for jmg in allimages:
                jmg = cv2.cvtColor(jmg, cv2.COLOR_BGR2RGB)              
                encode = face_recognition.face_encodings(jmg)[0]
                enList.append(encode)
                return enList


        enListKnowns = Encodings(allimages)   # marked and id the image in webcam
        print('Encoding Complete')
        
        
        cap = cv2.VideoCapture("TestVideo/Test-Video.mp4")
        cap.set(480,240)
        # initialize web cam and check
        while True:
            success, jmg = cap.read()
            
            imgs = cv2.resize(jmg, (0, 0), None, 0.25, 0.25)
            imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
            facesCurrentFrame = face_recognition.face_locations(imgs)
            encodesCurrentFrame = face_recognition.face_encodings(imgs, facesCurrentFrame)

            for enFace, faceLocation in zip(encodesCurrentFrame, facesCurrentFrame):  #check each image frame
                matching = face_recognition.compare_faces(enListKnowns, enFace)
                faceDistance = face_recognition.face_distance(enListKnowns, enFace)

                matchingIndex = np.argmin(faceDistance)

                if matching[matchingIndex]:  # marking the box over image if matched
                    naming = Names[matchingIndex].upper()

                    y1, x2, y2, x1 = faceLocation
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(jmg, (x1, y1), (x2, y2), (255, 0, 0), 2)  #image rectangle
                    cv2.rectangle(jmg, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED) #text rectangle
                    cv2.putText(jmg, naming, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    Attendance(naming)

            cv2.imshow('Detectors', jmg)
    
    
            if cv2.waitKey(1) == ord('q'):   # break loop by pressing q on keyboard
                break                
    
    
        cap.release()
        cv2.destroyAllWindows()    #Close the webcam window
