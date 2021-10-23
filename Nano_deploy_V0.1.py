from gtts import gTTS
import os
from playsound import playsound
import speech_recognition as sr
import wikipedia

import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

intro = '''Hello, I am Maya 2.0 created at the Department of Automation and Robotics in KLE Technological University. I am built to mimic human actions, interact with humans and to act as a medicine dispenser in hospitals.'''

department_intro = '''I was created in the Department of Automation and Robotics Engineering in KLE Technological University. Mr. Arun Giriyapur is the head of the department. BE Automation & Robotics program prepares students for designing, interface, installation, and troubleshooting of industrial robots and automation systems. The main emphasis is on cognitive intelligence, electronics, electrical controls, robotics and mechanical systems. Students integrate electronics and electrical controls with mechanical systems and programmable controllers and explore alternative trade-offs in the process of problem-solving.'''

def speak(text):
    tts = gTTS(text,lang = 'en',tld='co.in',slow=False)
    tts.save('answer.mp3')
    sound_file = 'answer.mp3'
    playsound(sound_file)
    print("Maya has spoken")

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def introduction():
    print("samajh aa gaya")
    speak(intro)

def department_introduction():
    print("samajh aa gaya")
    speak(department_intro)

def ask_wiki(ques):
    print("Searching...")
    page_data = wikipedia.page(ques)
    answer = page_data.content
    if answer is None:
        return -1
    else:
        sen = split_into_sentences(answer)
        return sen
def listen():
    r = sr.Recognizer()
    mic = sr.Microphone(device_index=9)
    print("Be ready to speak")
    with mic as source:
        r.adjust_for_ambient_noise(source)
        print("START SPEAKING NOW")
        audio = r.listen(source)

    str = r.recognize_google(audio)
    return str

while True:

    str = listen()
    str = str.split()
    if "introduce" in str or "introduction" in str:
        if "department" in str:
            department_introduction()
        else:
            introduction()
    elif "know" in str:
        speak('What would you like to know about')
        topic = listen()
        sen = ask_wiki(topic)
        if sen == -1:
            speak("Sorry, I don't know the answer")
        else:
            a = 0
            b = 2
            flag = True
            while flag:
                s = ""
                s = s.join(sen[a:b])
                speak(s)
                speak("Do you want to know more")
                choice = listen()
                if choice == "yes":
                    a = b
                    b+=5
                else:
                    flag = False
    elif "exit" in str:
        break
