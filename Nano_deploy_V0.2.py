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



intro = '''Hello, I am Maya 2.0 created at the Department of Automation and Robotics in KLE Technological University. I am built to mimic human actions, interact with humans and to act as a medicine dispenser in hospitals.'''

department_intro = '''I was created in the Department of Automation and Robotics Engineering in KLE Technological University. Mr. Arun Giriyapur is the head of the department. BE Automation & Robotics program prepares students for designing, interface, installation, and troubleshooting of industrial robots and automation systems. The main emphasis is on cognitive intelligence, electronics, electrical controls, robotics and mechanical systems. Students integrate electronics and electrical controls with mechanical systems and programmable controllers and explore alternative trade-offs in the process of problem-solving.'''


context = '''Executive council is led by Professor Ashok S. Shettar, the Vice-Chancellor.
Professor Ashok S. Shettar is the Vice-Chancellor.
Doctor P. G. Tewari is the academic dean.
Professor Uma K. Mudenagudi is the Research and Development dean.
C.A. Pooja R. Kandoi is the Finance officer.
Doctor Sanjay Kotabagi is the Student Welfare dean and Head of Humanities department.
Professor B. L. Desai is the Executive Dean.
Doctor B. B. Kotturshettar is the Dean Planning & Development.
Professor T. V. M. Swamy is the First Year Coordinator.
Professor C. D. Kerure is the Placement Officer
Doctor Vijayalakshmi M. is the Director of CEER.
Doctor Nitin Kulkarni is the Director of CTIE.

Heads of the departments :
Professor Arun C. Giriyapur is the Head of Automation and Robotics Engineering Department
Doctor Vinaya Hiremath is the Head of School of Architecture
Doctor B. S. Hunagund is the Head of Biotechnology Department
Doctor M. V. Chitawadagi is the Head of School of Civil Engineering
Doctor Meena S. Marallappanavar is the Head of School of Computer Science & Engineering
Doctor A. B. Raju is the Head of Electrical and Electronics Department
Doctor Nalini Iyer is the Head of School of Electronics & Communication Engineering
Professor P. R. Patil is the Head of MCA department
Professor Jagdish Bapat is the Director of School of Management Studies and Research
Doctor B. B. Kotturshettar is the Head of School of Mechanical Engineering
Professor S V Hiremath is the head of School of Advanced sciences
Professor G B Marali is the Head of Mathematics department
'''





use_own_model = False

model_name_or_path = "ktrapeznikov/albert-xlarge-v2-squad-v2"

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
    speak("What question do you want me to answer")
    ques = listen()
    prediction = run_prediction(ques, context)
    if prediction["0"] == None:
        speak("Sorry, I don't know the answer to that question")
    else:
        speak(prediction["0"])


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
    speak("Hand on a minute")
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

    speak("What do you want me to do")
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
    elif "ask" in str or "answer" in str:
        ask()
    elif "exit" in str:
        break
