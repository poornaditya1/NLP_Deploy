from gtts import gTTS
import os
from playsound import playsound
import speech_recognition as sr
import wikipedia
import pyjokes
import datetime
from gnewsclient import gnewsclient
import requests
from urllib.request import urlopen
import json

import os
import torch
import time
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import (
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    squad_convert_examples_to_features
)

from transformers.data.processors.squad import SquadResult, SquadV2Processor, SquadExample

from transformers.data.metrics.squad_metrics import compute_predictions_logits


import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"



intro = '''Hello, I am Maya 1.0 created at the Department of Automation and Robotics in KLE Technological University. I am built to mimic human actions and interact with humans. I can move, navigate and verbally interact with people around me'''

department_intro = '''I was created in the Department of Automation and Robotics Engineering in. KLE Technological University. The department prepares students to work efficiently with industrial robots and automation systems. The main areas of focus are cognitive intelligence, electronics, robotics and mechanical systems. Students integrate electronics with mechanical systems and programmable controllers and explore alternative trade-offs in the process of problem-solving. The students work on a variety of projects and create marvels of engineering like me'''

name = ''

use_own_model = False

model_name_or_path = "elgeish/cs224n-squad2.0-albert-large-v2" #"twmkn9/albert-base-v2-squad2"#"ktrapeznikov/albert-xlarge-v2-squad-v2"
output_dir = ""

n_best_size = 1
max_answer_length = 60
do_lower_case = True
null_score_diff_threshold = 0.0

def to_list(tensor):
    return tensor.detach().cpu().tolist()

# Setup model
config_class, model_class, tokenizer_class = (
    AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer)
config = config_class.from_pretrained(model_name_or_path)
tokenizer = tokenizer_class.from_pretrained(
    model_name_or_path, do_lower_case=True)
model = model_class.from_pretrained(model_name_or_path, config=config)

device = torch.device("cuda")

model.to(device)

processor = SquadV2Processor()

def run_prediction(question_texts, context_text):
    """Setup function to compute predictions"""
    examples = []

    example = SquadExample(
        qas_id="0",
        question_text=question_texts,
        context_text=context_text,
        answer_text=None,
        start_position_character=None,
        title="Predict",
        is_impossible=False,
        answers=None,
    )

    examples.append(example)

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=False,
        return_dataset="pt",
        threads=1,
    )

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=10)

    all_results = []

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]

            outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

    output_prediction_file = "predictions.json"
    output_nbest_file = "nbest_predictions.json"
    output_null_log_odds_file = "null_predictions.json"

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,  # verbose_logging
        True,  # version_2_with_negative
        null_score_diff_threshold,
        tokenizer,
    )

    return predictions

def ask():
    f = open("context.txt", "rt")
    context = f.read()
    f.close()
    w = open("context.txt", "at")

    # print("What question do you want me to answer")
    ques = listen(3)
    print("Question: ",ques)
    # ques = input()
    prediction = run_prediction(ques, context)
    print(prediction["0"])
    if prediction["0"] == '':
        speak("Sorry, I don't know the answer to that question")
        speak("Would you like to teach me the answer?")
        ch = listen()
        if ch=="yes":
            learn = listen(4)
            learn = learn+"."
            w.write(learn)
            speak("Thank you for teaching me that")
        print("Sorry, I don't know the answer to that question")
    else:
        print(prediction["0"])
        speak(prediction["0"])


def speak(text):
    tts = gTTS(text,lang = 'en',tld='co.in',slow=False)
    tts.save('answer.mp3')
    sound_file = 'answer.mp3'
    playsound(sound_file)
    print("Maya has spoken")
    os.remove('answer.mp3')
    #print(text)

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

# def ask_wiki(ques):
#     print("Searching...")
#     speak("Hand on a minute")
#     page_data = wikipedia.page(ques)
#     answer = page_data.content
#     if answer is None:
#         return -1
#     else:
#         sen = split_into_sentences(answer)
#         return sen

def ask_wiki(str):
    topic = ""
    flag = -1
    index = str.index("about")
    length = len(str)
    for i in range(index+1,length):
        topic = topic + str[i]
        if i is not length-1:
            topic = topic + " "

    print("Searching...")
    speak("Hang on a minute")
    print("Topic = ",topic)
    try:
        page_data = wikipedia.page(topic, auto_suggest=False)
        answer = page_data.content
    except wikipedia.DisambiguationError as e:
        speak("I am confused between the following options")
        for opt in e.options:
            speak(opt)
            print(opt)
        topic = listen(5)
        sen = ask_wiki(topic)
        return sen
    if answer is None:
        return -1
    else:
        sen = split_into_sentences(answer)
        return sen

def listen(a=None):
    r = sr.Recognizer()
    mic = sr.Microphone(device_index=26)
    print("Be ready to speak")
    with mic as source:
        r.adjust_for_ambient_noise(source, duration = 0.5)
        if a is 0:
            speak("What should i do for you")
        elif a is 1:
            speak("Do you want to know more")
        elif a is 3:
            speak("What question do you want me to answer")
        elif a is 4:
            speak("Please tell me the answer")
        elif a is 5:
            speak("Please select your choice")
        elif a is 6:
            speak("What is your name?")
        elif a is 7:
            speak("Please tell me your topic of interest")
        elif a is None:
            pass

        print("START SPEAKING NOW")
        audio = r.listen(source)

    try:
        string = r.recognize_google(audio)
    #string = input()
        return string
    except:
        speak("I'm sorry, I didn't get you. Please say it again")
        return listen()

def greet(name):
    cur_time = time.localtime(time.time())
    if cur_time.tm_hour<12:
        greeting = "Good morning "
    elif cur_time.tm_hour<17:
        greeting = "Good afternoon "
    else:
        greeting = "Good evening "
    return greeting+name


if __name__=='__main__':

    speak("Maya is online and ready to comply")
    name = listen(6)
    speak(greet(name))
    while True:

    # print("What do you want me to do")
        command = listen(0)
    # str = input()
        str_q = command.split()
        if "how are you" in command:
            speak("I'm doing great")
        elif "name" in str_q:
            speak("My name is Maya")    
        elif "introduce" in str_q or "introduction" in str_q:
            if "department" in str_q:
                department_introduction()
            else:
                introduction()
        elif "joke" in str_q:
            jk = pyjokes.get_joke("en")
            speak(jk)
            print(jk)

        elif 'news' in str_q:
            n_topic = listen(7)
            client = gnewsclient.NewsClient(language = 'english', location = 'india', topic = n_topic, max_results = 3)
            news = client.get_news()
            speak("The headlines are as follows")
            for item in news:
                speak(item['title'])
        elif "weather" in str_q:
            api_k = "4e68859684a3c41718d5aaa517641d1f"
            base_url = "http://api.openweathermap.org/data/2.5/weather?"
            city = "Hubli"
            complete_url = base_url + "appid=" + api_k + "&q=" + city
            response = requests.get(complete_url)
            x = response.json()
            if x["cod"] != "404":
                y = x["main"]
                current_temperature = y["temp"]
                current_pressure = y["pressure"]
                current_humidiy = y["humidity"]
                z = x["weather"]
                weather_description = z[0]["description"]
                speak(" Temperature (in kelvin unit) = " +str(current_temperature)+"\n atmospheric pressure (in hPa unit) ="+str(current_pressure) +"\n humidity (in percentage) = " +str(current_humidiy) +"\n description = " +str(weather_description))

        elif "about" in str_q:
            #speak('What would you like to know about')
            #topic = listen()
            # sen = ask_wiki(topic)
            sen = ask_wiki(str_q)
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

                    choice = listen(1)
                    if choice == "yes":
                        a = b
                        b+=5
                    else:
                        flag = False
        elif "ask" in str_q or "answer" in str_q:
            ask()
        elif "exit" in str_q or "bye" in str_q:
            cont = "thank you "+name
            if time.localtime(time.time()).tm_hour < 20:
                speak(cont+". Have a great day")
            else:
                speak(cont+". Good night")
            name =''
            break
