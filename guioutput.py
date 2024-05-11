import tkinter as tk, threading
from tkinter import ttk
from PIL import ImageTk, Image
import cv2
import mediapipe as mp
import time
import os
import numpy as np
import glob
import tensorflow as tf
import re
from collections import Counter
import math
import random
import imageio

sequence = []
sentence = []
sentences = []
sentences1 = []
pp = ""
with open("sentencesselected.txt","r") as f:
    for i in f:
        print(i.split("\t")[0].replace("\n",""))
        # k = i.split("\t")[0].replace("\n","")
        sentences.append(i.split("\t")[1].replace("\n",""))
        sentences1.append(i.split("\t")[0].replace("\n",""))

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

WORD = re.compile(r"\w+")
sen = [text_to_vector(i) for i in sentences]
model = tf.keras.models.load_model('lstm1.h5')
# interpreter = tf.lite.Interpreter(model_path="lstmsmall0124.tflite")
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# print(interpreter.get_input_details())
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
a = []
with open("wordsselected.txt","r") as f:
    for i in f:
        a.append(i.replace("\n",""))
# print(a)
DATA_PATH = os.path.join('MP_Data')
actions = np.array([i for i in sorted(a)])
predictions = []
threshold = 0.5

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    #return np.concatenate([pose, face, lh, rh])
    return np.concatenate([pose, lh, rh])

LARGEFONT =("Verdana", 35)

class tkinterApp(tk.Tk):
    
    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
        
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)
        
        # creating a container
        # container = tk.Frame(self, width=800, height=600)
        container = tk.Frame(self, width=800, height=600)
        container.pack(side = "top", fill = "both", expand = True)
        # container.pack_propagate(0)

        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)
        self.container=container

        # initializing frames to an empty array
        # self.frames = {}

        # iterating through a tuple consisting
        # of the different page layouts
        # for F in (Welcome, Page1, Page2):

        #     frame = F(container, self)

        #     # initializing frame of that object from
        #     # Welcome, page1, page2 respectively with
        #     # for loop
        #     self.frames[F] = frame

        #     frame.grid(row = 0, column = 0, sticky ="nsew")
        #     # frame.grid_propagate(1)
        #     frame.place(width=800,height=800)
        self.frames = {
                        "Welcome": Welcome, 
                        "Page1": Page1, 
                        "Page2": Page2,
                    }

        self.show_frame("Welcome")

    # to display the current frame passed as
    # parameter
    # def show_frame(self, cont):
    #     frame = self.frames[cont]
    #     frame.tkraise()
    #     frame.event_generate("<<ShowFrame>>")
    def show_frame(self, page_name):
        # destroy the old frame
        # print(self.container.winfo_children())
        for child in self.container.winfo_children():
            # child.grid_forget()
            # child.quit()
            child.destroy()
        # print(self.container.winfo_children())
        # create the new frame
        # print(page_name)
        frame_class = self.frames[page_name]
        frame = frame_class(self.container, self)
        frame.grid(row = 0, column = 0, sticky ="nsew")
        frame.place(width=800,height=600)
        frame.tkraise()
        # frame.pack(side = "top", fill="both", expand=True)

# first window frame Welcome

class Welcome(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        img = ImageTk.PhotoImage(Image.open("img.png").resize((800, 600), Image.ANTIALIAS))
        lbl = tk.Label(self, image=img)
        lbl.img = img
        lbl.place(relx=0.5, rely=0.5, anchor='center')
        
        # label of frame Layout 2
        label = ttk.Label(self, text ="Welcome", font = LARGEFONT)
        
        # putting the grid in its place by using
        # grid
        label.grid(row = 0, column = 1, padx = 10, pady = 50)

        button1 = ttk.Button(self, text ="Sentence to video",
        command = lambda : controller.show_frame("Page1"))
    
        # putting the button in its place by
        # using grid
        button1.grid(row = 1, column = 1, padx = 10, pady = 50)

        ## button to show frame 2 with text layout2
        button2 = ttk.Button(self, text ="Video to sentence",
        command = lambda : controller.show_frame("Page2"))
    
        # putting the button in its place by
        # using grid
        button2.grid(row = 2, column = 1, padx = 10, pady = (10,150))
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        


# second window frame page1
# class Page1(tk.Frame):
    
#     def __init__(self, parent, controller):
        
#         tk.Frame.__init__(self, parent)
#         img = ImageTk.PhotoImage(Image.open("img.png").resize((800, 600), Image.ANTIALIAS))
#         lbl = tk.Label(self, image=img)
#         lbl.img = img
#         lbl.place(relx=0.5, rely=0.5, anchor='center')
#         label = ttk.Label(self, text ="Sentence to video", font = LARGEFONT)
#         label.grid(row = 0, column = 4, padx = 10, pady = 10)

#         # button to show frame 2 with text
#         # layout2
#         button1 = ttk.Button(self, text ="Welcome",
#                             command = lambda : controller.show_frame("Welcome"))
    
#         # putting the button in its place
#         # by using grid
#         button1.grid(row = 1, column = 1, padx = 10, pady = 10)

#         # button to show frame 2 with text
#         # layout2
#         button2 = ttk.Button(self, text ="Video to sentence",
#                             command = lambda : controller.show_frame("Page2"))
    
#         # putting the button in its place by
#         # using grid
#         button2.grid(row = 2, column = 1, padx = 10, pady = 10)

# cap = None
class Page1(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        img = ImageTk.PhotoImage(Image.open("img.png").resize((800, 600), Image.ANTIALIAS))
        lbl = tk.Label(self, image=img)
        lbl.img = img
        lbl.place(relx=0.5, rely=0.5, anchor='center')
        label = ttk.Label(self, text ="Sentence to video", font = LARGEFONT)
        label.grid(row = 0, column = 4, padx = 10, pady = 10)

        # button to show frame 2 with text
        # layout2
        button1 = ttk.Button(self, text ="Welcome",
                            command = lambda : controller.show_frame("Welcome"))
    
        # putting the button in its place
        # by using grid
        button1.grid(row = 1, column = 1, padx = 10, pady = 10)

        # button to show frame 2 with text
        # layout2
        button2 = ttk.Button(self, text ="Video to sentence",
                            command = lambda : controller.show_frame("Page2"))
    
        # putting the button in its place by
        # using grid
        button2.grid(row = 2, column = 1, padx = 10, pady = 10)

        lmain = tk.Label(self)
        lmain.grid(row=5, column=3, padx = 5, pady = 10,columnspan = 3, rowspan = 2)
        l = ttk.Label(self, text = "Enter sentence:")
        l.place(x=350,y=100)
        l.config(font =("Courier", 14))
        T = tk.Text(self, height = 5, width = 50)
        T.place(x=350,y=120)
        # thread = threading.Thread(target=kk1, args=(lmain,T))
        # thread.daemon = 1
        # thread.start()
        # p = T.get(1.0,tk.END).strip()
        # if p!="":
        #     vid = glob.glob(f"Videos_Sentence_Level/{p}/*")[random.randint(0,len(glob.glob(f"Videos_Sentence_Level/{p}/*")))]
        #     cap = cv2.VideoCapture(vid)
        # cap = cv2.VideoCapture('Videos_Sentence_Level/are you free today/are you free today.mp4')

        def kk1(cap):
            try:
                ret, frame = cap.read()
                frame = cv2.resize(frame, (480, 360))  

                cv2image   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                ksize = (2, 2)
                cv2image = cv2.blur(cv2image, ksize) 

                img   = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image = img)
                lmain.imgtk = imgtk
                lmain.configure(image=imgtk)
                lmain.after(10, kk1,cap)
            except Exception as e:
                pass

        def kk():
            global cap
            cap.release()
            cv2.destroyAllWindows()

        def kk11():
            p = T.get(1.0,tk.END).strip()
            if p!="":
                try:
                    vid = glob.glob(f"Videos_Sentence_Level/{p}/*")[random.randint(0,len(glob.glob(f"Videos_Sentence_Level/{p}/*")))]
                    cap = cv2.VideoCapture(vid)
                    kk1(cap)
                except Exception as e:
                    pass
            # print(T.get(1.0,tk.END).strip())

        # def kk1(lmain,T):
        #     try:
        #         p = T.get(1.0,tk.END).strip()
        #         vid = glob.glob(f"Videos_Sentence_Level/{p}/*")[random.randint(0,len(glob.glob(f"Videos_Sentence_Level/{p}/*")))]
        #         video = imageio.get_reader(vid)
        #         for image in video.iter_data():
        #             frame_image = ImageTk.PhotoImage(Image.fromarray(image))
        #             lmain.config(image=frame_image)
        #             lmain.image = frame_image
        #     # vc = cv2.VideoCapture(vid)
        #     # if vc.isOpened():
        #     #     rval , frame = vc.read()
        #     # else:
        #     #     rval = False

        #     # while rval:
        #     #     rval, frame = vc.read()
        #     #     frame = cv2.resize(frame, (480, 360))  
        #     #     cv2.imshow("kk",frame)
        #     #     # frame = cv2.flip(frame, 1)
        #     #     cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        #     #     img =Image.fromarray(cv2image)
        #     #     # img = resize(img)
        #     #     imgtk = ImageTk.PhotoImage(img)
        #     #     lmain.config(image=imgtk)
        #     #     lmain.img = imgtk
        #     #     # if stop == True: 
        #     #     #     vc.release()
        #     #     #     break      #stop the loop thus stops updating the label and reading imagge frames
        #     #     # cv2.waitKey(0)
        #     # vc.release()
        #     # p = T.get(1.0,tk.END).strip()
        #     # vid = glob.glob(f"Videos_Sentence_Level/{p}/*")[random.randint(0,len(glob.glob(f"Videos_Sentence_Level/{p}/*")))]
        # #     print(vid)
        # #     cap = cv2.VideoCapture(vid)
               
        # #     # Check if camera opened successfully
        # #     if (cap.isOpened()== False):
        # #         print("Error opening video  file")
               
        # #     # Read until video is completed
        # #     while(cap.isOpened()):
        # #     # Capture frame-by-frame
        # #         ret, frame = cap.read()
        # #         if ret == True:
                   
        # #             # Display the resulting frame
        # #             frame = cv2.resize(frame, (480, 360))  
        # #             cv2.imshow("kk",frame)
        # #             # frame = cv2.flip(frame, 1)
        # #             cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        # #             img = Image.fromarray(cv2image)
        # #             imgtk = ImageTk.PhotoImage(image=img)
        # #             lmain.imgtk = imgtk
        # #             lmain.configure(image=imgtk)
        # #             # lmain.after(10, kk1, vid)
                   
        # #             # Press Q on keyboard to  exit
        # #             # if cv2.waitKey(25) & 0xFF == ord('q'):
        # #             #     break
               
        # #     # else: 
        # #     #     break
               
        # #     # When everything done, release 
        # #     # the video capture object
        # #     # cap.release()
               
        # #     # # Closes all the frames
        # #     # cv2.destroyAllWindows()
        #     except Exception as e:
        #         pass


        def kk2(T):
            T.delete(1.0, tk.END)

        # button3 = ttk.Button(self, text ="close video",
        #                     command = lambda : kk())
        # button3.grid(row = 3, column = 1, padx = 10, pady = 10)

        button4 = ttk.Button(self, text ="Play sentence video",
                            command = lambda : kk11())
                            # command = lambda : kk1(cv2.VideoCapture(glob.glob(f"Videos_Sentence_Level/{T.get(1.0,tk.END).strip()}/*")[random.randint(0,len(glob.glob(f"Videos_Sentence_Level/{T.get(1.0,tk.END).strip()}/*")))])))
        button4.grid(row = 3, column = 1, padx = 10, pady = 10)

        button5 = ttk.Button(self, text ="clear output",
                            command = lambda : kk2(T))
        button5.grid(row = 3, column = 2, padx = 10, pady = 10)
    
        # putting the button in its place by
        # using grid
        #Capture video frames
        # lmain = tk.Label(self)
        # lmain.grid(row=5, column=3, padx = 5, pady = 10,columnspan = 3, rowspan = 2)
        # l = ttk.Label(self, text = "Enter sentence:")
        # l.place(x=350,y=100)
        # l.config(font =("Courier", 14))
        # T = tk.Text(self, height = 5, width = 50)
        # T.place(x=350,y=120)
        # if len(pred)>0:
        #     T.insert(tk.END, pred)
        # lmain.place(width=400,height=400)
        # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # def show_frame1(T):
        #     global sequence
        #     global sentence
        #     global pp
        #     try:
        #         _, frame = cap.read()
        #         image, results = mediapipe_detection(frame, holistic)
        #         # print(results)
        #         draw_styled_landmarks(image, results)
        #         keypoints = extract_keypoints(results)
        #         sequence.append(keypoints)
        #         # sequence = sequence[-30:]
        #         # print(len(sequence))
                
        #         if len(sequence) == 30:
        #             res = model.predict(np.expand_dims(sequence, axis=0))[0]
        #             predictions.append(np.argmax(res))
            
        #             # if np.unique(predictions[-10:])[0]==np.argmax(res): 
        #             if results.left_hand_landmarks or results.right_hand_landmarks:
        #                 if res[np.argmax(res)] > threshold: 
                            
        #                     if len(sentence) > 0: 
        #                         if actions[np.argmax(res)] != sentence[-1]:
        #                             sentence.append(actions[np.argmax(res)])
        #                     else:
        #                         sentence.append(actions[np.argmax(res)])

        #             if len(sentence) > 5: 
        #                 sentence = sentence[-5:]
        #             if len(sentence) > 0:
        #                 vector1 = text_to_vector(" ".join(sentence))
        #                 # print(sentences[np.argmax([get_cosine(vector1, text_to_vector(i)) for i in sentences])])
        #                 pp = sentences1[np.argmax([get_cosine(vector1, i) for i in sen])]
        #                 # print([get_cosine(vector1, i) for i in sen])
        #                 # print("sen:",sentence)
        #                 # print("pp:",pp)
        #                 # if len(pp)>0:
        #                 #     T.delete(1.0, tk.END)
        #                 #     T.insert(tk.END, pp)
        #             sequence = []
        #             # time.sleep(1.5)

        #             pred = [j for j in [actions[i] for i in ((-res).argsort()[:5])]]
        #             # print(pred)
        #         # cv2.imshow("k",frame)
        #         cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        #         cv2.putText(image, ' '.join(sentence), (3,30), 
        #                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #         # print(frame.shape)
        #         frame = cv2.resize(image, (480, 360))  
        #         # frame = cv2.flip(frame, 1)
        #         cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        #         img = Image.fromarray(cv2image)
        #         imgtk = ImageTk.PhotoImage(image=img)
        #         lmain.imgtk = imgtk
        #         lmain.configure(image=imgtk)
        #         lmain.after(10, show_frame1, T)
        #         # cap.release()
        #         # cv2.destroyAllWindows()
        #     except Exception as e:
        #         # print(e)
        #         pass

        # if T.get(1.0,tk.END).strip()!="":
        # thread = threading.Thread(target=kk1, args=(lmain,T))
        # thread.daemon = 1
        # thread.start()
            # print("T",T.get(1.0,tk.END).strip())
            # kk1(T)

# third window frame page2
class Page2(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        img = ImageTk.PhotoImage(Image.open("img.png").resize((800, 600), Image.ANTIALIAS))
        lbl = tk.Label(self, image=img)
        lbl.img = img
        lbl.place(relx=0.5, rely=0.5, anchor='center')
        # self.bind("<<ShowFrame>>", self.show_frame)
        label = ttk.Label(self, text ="Video to sentence", font = LARGEFONT)
        label.grid(row = 0, column = 4, padx = 10, pady = 10)

        # def kk(cap):
        #     cap.release()
        #     cv2.destroyAllWindows()

        # # button to show frame 2 with text
        # # layout2
        # button1 = ttk.Button(self, text ="Sentence to video",
        #                     command = lambda : kk(cap))
        button1 = ttk.Button(self, text ="Sentence to video",
                            command = lambda : controller.show_frame("Page1"))
    
        # putting the button in its place by
        # using grid
        button1.grid(row = 1, column = 1, padx = 10, pady = 10)

        # button to show frame 3 with text
        # layout3
        button2 = ttk.Button(self, text ="Welcome",
                            command = lambda : controller.show_frame("Welcome"))
        button2.grid(row = 2, column = 1, padx = 10, pady = 10)

        def kk(cap):
            cap.release()
            cv2.destroyAllWindows()

        def kk1(T):
            if len(pp)>0:
                T.insert(tk.END, pp)

        def kk2(T):
            global sentence
            if len(pp)>0:
                sentence = []
                T.delete(1.0, tk.END)

        def kk3():
            global sentence
            if len(pp)>0:
                sentence = [i for i in sentence[:-1]]
                # T.delete(1.0, tk.END)

        button3 = ttk.Button(self, text ="close video",
                            command = lambda : kk(cap))
        button3.grid(row = 3, column = 1, padx = 10, pady = 10)

        button4 = ttk.Button(self, text ="predict sentence",
                            command = lambda : kk1(T))
        button4.grid(row = 3, column = 2, padx = 10, pady = 10)

        button5 = ttk.Button(self, text ="clear output",
                            command = lambda : kk2(T))
        button5.grid(row = 3, column = 3, padx = 10, pady = 10)

        button6 = ttk.Button(self, text ="clear word",
                            command = lambda : kk3())
        button6.grid(row = 4, column = 1, padx = 10, pady = 10)
    
        # putting the button in its place by
        # using grid
        #Capture video frames
        lmain = tk.Label(self)
        lmain.grid(row=5, column=3, padx = 5, pady = 10,columnspan = 3, rowspan = 2)
        l = ttk.Label(self, text = "predicted sentence:")
        l.place(x=350,y=100)
        l.config(font =("Courier", 14))
        T = tk.Text(self, height = 5, width = 50)
        T.place(x=350,y=120)
        # if len(pred)>0:
        #     T.insert(tk.END, pred)
        # lmain.place(width=400,height=400)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        def show_frame1(T):
            global sequence
            global sentence
            global pp
            try:
                _, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                # print(results)
                draw_styled_landmarks(image, results)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                # sequence = sequence[-30:]
                # print(len(sequence))
                
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predictions.append(np.argmax(res))
            
                    # if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if results.left_hand_landmarks or results.right_hand_landmarks:
                        if res[np.argmax(res)] > threshold: 
                            
                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]
                    if len(sentence) > 0:
                        vector1 = text_to_vector(" ".join(sentence))
                        # print(sentences[np.argmax([get_cosine(vector1, text_to_vector(i)) for i in sentences])])
                        pp = sentences1[np.argmax([get_cosine(vector1, i) for i in sen])]
                        # print([get_cosine(vector1, i) for i in sen])
                        # print("sen:",sentence)
                        # print("pp:",pp)
                        # if len(pp)>0:
                        #     T.delete(1.0, tk.END)
                        #     T.insert(tk.END, pp)
                    sequence = []
                    time.sleep(0.5)

                    pred = [j for j in [actions[i] for i in ((-res).argsort()[:5])]]
                    # print(pred)
                # cv2.imshow("k",frame)
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # print(frame.shape)
                frame = cv2.resize(image, (480, 360))  
                # frame = cv2.flip(frame, 1)
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                lmain.imgtk = imgtk
                lmain.configure(image=imgtk)
                lmain.after(10, show_frame1, T)
                # cap.release()
                # cv2.destroyAllWindows()
            except Exception as e:
                # print(e)
                pass


        show_frame1(T)
        # if cv2.EVENT_RBUTTONDOWN:
        #     cap.release()
        # cv2.destroyAllWindows()


# Driver Code
app = tkinterApp()
app.mainloop()
