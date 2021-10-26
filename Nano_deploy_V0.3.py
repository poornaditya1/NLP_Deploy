from gtts import gTTS
import os
from playsound import playsound
import speech_recognition as sr
import wikipedia

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
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"



intro = '''Hello, I am Maya 1.0 created at the Department of Automation and Robotics in KLE Technological University. I am built to mimic human actions, interact with humans and to act as a medicine dispenser in hospitals.'''

department_intro = '''I was created in the Department of Automation and Robotics Engineering in. KLE Technological University. Mr. Arun Giriyapur is the head of the department. The department prepares students to work efficiently with industrial robots and automation systems. The main areas of focus are cognitive intelligence, electronics, robotics and mechanical systems. Students integrate electronics with mechanical systems and programmable controllers and explore alternative trade-offs in the process of problem-solving.'''


use_own_model = False

model_name_or_path = "twmkn9/albert-base-v2-squad2"#"ktrapeznikov/albert-xlarge-v2-squad-v2"
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    page_data = wikipedia.page(topic)
    answer = page_data.content
    if answer is None:
        return -1
    else:
        sen = split_into_sentences(answer)
        return sen

def listen(a=None):
    r = sr.Recognizer()
    mic = sr.Microphone(device_index=9)
    print("Be ready to speak")
    with mic as source:
        r.adjust_for_ambient_noise(source)
        if a is 0:
            speak("What should i do for you")
        elif a is 1:
            speak("Do you want to know more")
        elif a is 3:
            speak("What question do you want me to answer")
        elif a is 4:
            speak("Please tell me the answer")
        elif a is None:
            pass

        print("START SPEAKING NOW")
        audio = r.listen(source)

    str = r.recognize_google(audio)
    return str

speak("Maya is online and ready to comply")
while True:

    # print("What do you want me to do")
    str = listen(0)
    # str = input()
    str = str.split()
    if "introduce" in str or "introduction" in str:
        if "department" in str:
            department_introduction()
        else:
            introduction()
    elif "about" in str:
        #speak('What would you like to know about')
        #topic = listen()
        # sen = ask_wiki(topic)
        sen = ask_wiki(str)
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
    elif "ask" in str or "answer" in str:
        ask()
    elif "exit" in str:
        speak("Thank you")
        break
