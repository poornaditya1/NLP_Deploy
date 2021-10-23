from gtts import gTTS
import os
from playsound import playsound
import speech_recognition as sr

intro = '''Hello, I am Maya 2.0 created at the Department of Automation and Robotics in KLE Technological University. I am built to mimic human actions, interact with humans and to act as a medicine dispenser in hospitals.'''

department_intro = '''I was created in the Department of Automation and Robotics Engineering in KLE Technological University. Mr. Arun Giriyapur is the head of the department. BE Automation & Robotics program prepares students for designing, interface, installation, and troubleshooting of industrial robots and automation systems. The main emphasis is on cognitive intelligence, electronics, electrical controls, robotics and mechanical systems. Students integrate electronics and electrical controls with mechanical systems and programmable controllers and explore alternative trade-offs in the process of problem-solving.'''

def introduction(self):
    print("samajh aa gaya")
    tts = gTTS(intro, lang='en', tld='co.in',slow=False)
    tts.save('answer.mp3')
    sound_file = 'answer.mp3'
    playsound(sound_file)
    print("Maya has spoken")

def department_introduction(self):
    print("samajh aa gaya")
    tts = gTTS(department_intro, lang='en', tld='co.in',slow=False)
    tts.save('answer.mp3')
    sound_file = 'answer.mp3'
    playsound(sound_file)
    print("Maya has spoken")

while True:
    r = sr.Recognizer()
    mic = sr.Microphone(device_index=9)
    print("Be ready to speak")
    with mic as source:
        r.adjust_for_ambient_noise(source)
        print("START SPEAKING NOW")
        audio = r.listen(source)

    str = r.recognize_google(audio)
    str = str.split()
    if "introduce" in str or "introduction" in str:
        if "department" in str:
            department_introduction()
        else:
            introduction()
