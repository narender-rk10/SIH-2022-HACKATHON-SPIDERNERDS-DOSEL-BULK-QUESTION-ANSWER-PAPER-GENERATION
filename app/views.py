import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from django.shortcuts import render

from .models import DOSEL, MCQ, Board, BoardUsers, Subject, Topic,Users
from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from django.conf import settings
from django.contrib import messages
from django.core.mail import send_mail
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Create your views here.


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_boolean_questions')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print ("device ",device)
model = model.to(device)


def greedy_decoding(inp_ids, attn_mask):
    greedy_output = model.generate(
        input_ids=inp_ids, attention_mask=attn_mask, max_length=256)
    Question = tokenizer.decode(
        greedy_output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return Question.strip().capitalize()


def beam_search_decoding(inp_ids, attn_mask):
    beam_output = model.generate(input_ids=inp_ids,
                                 attention_mask=attn_mask,
                                 max_length=256,
                                 num_beams=10,
                                 num_return_sequences=3,
                                 no_repeat_ngram_size=2,
                                 early_stopping=True
                                 )
    Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in
                 beam_output]
    return [Question.strip().capitalize() for Question in Questions]


def topkp_decoding(inp_ids, attn_mask):
    topkp_output = model.generate(input_ids=inp_ids,
                                  attention_mask=attn_mask,
                                  max_length=256,
                                  do_sample=True,
                                  top_k=40,
                                  top_p=0.80,
                                  num_return_sequences=3,
                                  no_repeat_ngram_size=2,
                                  early_stopping=True
                                  )
    Questions = [tokenizer.decode(
        out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in topkp_output]
    return [Question.strip().capitalize() for Question in Questions]

def dosel_login_required(function):
    def wrapper(request, *args, **kw):
        u_type = request.session['u_type']
        if u_type != 'dosel' and 'd_id' not in request.session:
            return HttpResponseRedirect('/')
        else:
            return function(request, *args, **kw)
    return wrapper

def board_login_required(function):
    def wrapper(request, *args, **kw):
        u_type = request.session['u_type']
        if u_type != 'board' and 'b_id' not in request.session:
            return HttpResponseRedirect('/')
        else:
            return function(request, *args, **kw)
    return wrapper



def teacher_login_required(function):
    def wrapper(request, *args, **kw):
        u_type = request.session['u_type']
        if u_type != 'teacher' and 't_id' not in request.session:
            return HttpResponseRedirect('/')
        else:
            return function(request, *args, **kw)
    return wrapper

def contributor_login_required(function):
    def wrapper(request, *args, **kw):
        u_type = request.session['u_type']
        if u_type != 'contributor' and 'c_id' not in request.session:
            return HttpResponseRedirect('/')
        else:
            return function(request, *args, **kw)
    return wrapper

def experts_login_required(function):
    def wrapper(request, *args, **kw):
        u_type = request.session['u_type']
        if u_type != 'experts' and 'e_id' not in request.session:
            return HttpResponseRedirect('/')
        else:
            return function(request, *args, **kw)
    return wrapper

def index(request):
    return render(request, 'index.html')

@dosel_login_required
def dosel_index(request):
    return render(request, 'DoselIndex.html')

@board_login_required
def board_index(request):
    return render(request, 'BoardIndex.html')

@teacher_login_required
def teacher_index(request):
    return render(request, 'TeacherIndex.html')

@contributor_login_required
def contributor_index(request):
    return render(request, 'ContributorIndex.html')

@experts_login_required
def experts_index(request):
    return render(request, 'ExpertsIndex.html')

def dosel_login(request):
    if request.method == "POST":
        o_email = request.POST['o_email']
        o_pass = request.POST['o_pass']
        dosel_details = DOSEL.objects.filter(email=o_email, password=o_pass).values()
        if dosel_details:
            request.session['logged_in'] = True
            request.session['d_email'] = dosel_details[0]["email"]
            request.session['d_id'] = dosel_details[0]["id"]
            request.session['d_name'] = dosel_details[0]["name"]
            request.session['u_type'] = "dosel"
            return HttpResponseRedirect('/dosel_index')
        else:
            return render(request, 'DoselLogin.html', {'details': "0"})
    else:
        return render(request, 'DoselLogin.html')

def board_login(request):
    if request.method == "POST":
        o_email = request.POST['o_email']
        o_pass = request.POST['o_pass']
        board_details = BoardUsers.objects.filter(email=o_email, password=o_pass).values()
        if board_details:
            request.session['logged_in'] = True
            request.session['b_email'] = board_details[0]["email"]
            request.session['b_id'] = board_details[0]["id"]
            request.session['u_type'] = "board"
            return HttpResponseRedirect('/board_index')
        else:
            return render(request, 'BoardLogin.html', {'details': "0"})
    else:
        return render(request, 'BoardLogin.html')

@dosel_login_required
def add_board(request):
    if request.method == "POST":
        b_name = request.POST['b_name']
        board_details = Board.objects.create(name=b_name)
        if board_details:
            return render(request, 'AddBoard.html', {'msg': "1"})
        else:
            return render(request, 'AddBoard.html', {'msg': "0"})
    return render(request, 'AddBoard.html')

@dosel_login_required
def read_boards(request):
    if request.method == 'GET':
        board_details = Board.objects.values()
        return render(request, 'ViewBoard.html', {"msg": board_details})

@dosel_login_required
def add_board_users(request):
    if request.method == "POST":
        b_id = request.POST['b_id']
        email = request.POST['email']
        password = '123456'
        contact = request.POST['contact']
        board_users_details = BoardUsers.objects.create(b_id_id=b_id,email=email,password=password,contact=contact)
        if board_users_details:
            return render(request, 'AddBoardUser.html', {'msg': "1"})
        else:
            return render(request, 'AddBoardUser.html', {'msg': "0"})
    else:
        board = Board.objects.values()
        return render(request, 'AddBoardUser.html', {"msg1": board})

@dosel_login_required
def read_boards_users(request):
    if request.method == 'GET':
        board_users_details = BoardUsers.objects.values()
        return render(request, 'ViewBoardUsers.html', {"msg": board_users_details})

@board_login_required
def add_sub(request):
    if request.method == "POST":
        b_id = request.session['b_id']
        s_name = request.POST['s_name']
        subject_details = Subject.objects.create(b_id_id=b_id,s_name=s_name)
        if subject_details:
            return render(request, 'AddSubject.html', {'msg': "1"})
        else:
            return render(request, 'AddSubject.html', {'msg': "0"})
    return render(request, 'AddSubject.html')


@board_login_required
def read_subjects(request):
    if request.method == 'GET':
        b_id = request.session['b_id']
        subject_details = Subject.objects.filter(b_id_id=b_id).values()
        return render(request, 'ViewSubjects.html', {"msg": subject_details})

@board_login_required
def add_topic(request):
    b_id = request.session['b_id']
    if request.method == "POST":
        s_id = request.POST['s_id']
        topic = request.POST['topic']
        topic_details = Topic.objects.create(b_id_id=b_id, s_id_id=s_id, t_name=topic)
        if topic_details:
            return render(request, 'AddTopic.html', {'msg': "1"})
        else:
            return render(request, 'AddTopic.html', {'msg': "0"})
    else:
        subjects = Subject.objects.filter(b_id_id=b_id).values()
        return render(request, 'AddTopic.html', {"msg1": subjects})

@board_login_required
def read_topic(request):
    if request.method == 'GET':
        b_id = request.session['b_id']
        topic_details = Topic.objects.filter(b_id_id=b_id).values()
        return render(request, 'ViewTopic.html', {"msg": topic_details})

@csrf_exempt
def logout(request):
    if request.method == 'POST':
        try:
            for key in list(request.session.keys()):
                del request.session[key]
            return render(request, 'index.html', {"msg": "1"})
        except:
            print("error")
        return render(request, 'index.html', {"msg": "0"})


def add_users(request):
    if request.method == "POST":
        b_id = request.session['b_id']
        name = request.POST['name']
        email = request.POST['email']
        password = '123456'
        user_type = request.POST['user_type']
        contact = request.POST['contact']
        users_details = Users.objects.create(b_id_id=b_id,email=email,password=password,contact=contact,user_type=user_type,name=name)
        if users_details:
            return render(request, 'AddUserByBoard.html', {'msg': "1"})
        else:
            return render(request, 'AddUserByBoard.html', {'msg': "0"})
    else:
        users = Users.objects.values()
        return render(request, 'AddUserByBoard.html', {"msg1": users})


def read_users(request):
    if request.method == 'GET':
        users_details = Users.objects.values()
        return render(request, 'ReadUserByBoard.html', {"msg": users_details})


#toChannge


def teacher_login(request):
    if request.method == "POST":
        u_email = request.POST['u_email']
        u_pass = request.POST['u_pass']
        user_details = Users.objects.filter(email=u_email, password=u_pass).values()
        if user_details[0]['user_type'] == '3':
            request.session['logged_in'] = True
            request.session['u_email'] = user_details[0]["email"]
            request.session['t_id'] = user_details[0]["id"]
            request.session['b_id'] = user_details[0]["b_id_id"]
            request.session['u_name'] = user_details[0]["name"]
            request.session['u_type'] = "teacher"
            return HttpResponseRedirect('/teacher_index')
        else:
            return render(request, 'TeacherLogin.html', {'details': "0"})
    else:
        return render(request, 'TeacherLogin.html')



def expert_login(request):
    if request.method == "POST":
        u_email = request.POST['u_email']
        u_pass = request.POST['u_pass']
        user_details = Users.objects.filter(email=u_email, password=u_pass).values()
        if user_details[0]['user_type'] == '4':
            request.session['logged_in'] = True
            request.session['u_email'] = user_details[0]["email"]
            request.session['u_name'] = user_details[0]["name"]
            request.session['e_id'] = user_details[0]["id"]
            request.session['b_id'] = user_details[0]["b_id_id"]
            request.session['u_type'] = "experts"
            return HttpResponseRedirect('/experts_index')
        else:
            return render(request, 'ExpertLogin.html', {'details': "0"})
    else:
        return render(request, 'ExpertLogin.html')



def contributor_login(request):
    if request.method == "POST":
        u_email = request.POST['u_email']
        u_pass = request.POST['u_pass']
        user_details = Users.objects.filter(email=u_email, password=u_pass).values()
        if user_details[0]['user_type'] == '5':
            request.session['logged_in'] = True
            request.session['u_email'] = user_details[0]["email"]
            request.session['c_id'] = user_details[0]["id"]
            request.session['u_name'] = user_details[0]["name"]
            request.session['b_id'] = user_details[0]["b_id_id"]
            request.session['u_type'] = "contributor"
            return HttpResponseRedirect('/contributor_index')
        else:
            return render(request, 'ContributorLogin.html', {'details': "0"})
    else:
        return render(request, 'ContributorLogin.html')

@teacher_login_required
def create_mcq(request):
    b_id = request.session['b_id']
    if request.method == "POST":
        u_id = request.session['t_id']
        s_id = request.POST['s_id']
        topic_id = request.POST['topic_id']
        question = request.POST['question']
        op1 = request.POST['op1']
        op2 = request.POST['op2']
        op3 = request.POST['op3']
        op4 = request.POST['op4']
        exp = request.POST['explanation']
        c_ans = request.POST['correct_ans']
        level = request.POST['level']
        status = '2'
        mcq_details = MCQ.objects.create(u_id_id=u_id, b_id_id=b_id, s_id_id=s_id, t_id_id=topic_id, question=question, option_1=op1, option_2=op2, option_3=op3, option_4=op4, correct_ans=c_ans, explanation=exp, status= status, level=level)
        if mcq_details:
            return render(request, 'createMCQ.html', {'msg': "1"})
        else:
            return render(request, 'createMCQ.html', {'msg': "0"})
    else:
        subject_details = Subject.objects.filter(b_id_id=b_id).values()
        topic_details = Topic.objects.filter(b_id_id=b_id).values()
        return render(request, 'createMCQ.html', {"msg1": subject_details, "msg2": topic_details})

@contributor_login_required
def create_mcq_contributor(request):
    b_id = request.session['b_id']
    if request.method == "POST":
        u_id = request.session['c_id']
        s_id = request.POST['s_id']
        topic_id = request.POST['topic_id']
        question = request.POST['question']
        op1 = request.POST['op1']
        op2 = request.POST['op2']
        op3 = request.POST['op3']
        op4 = request.POST['op4']
        exp = request.POST['explanation']
        c_ans = request.POST['correct_ans']
        level = request.POST['level']
        status = '2'
        mcq_details = MCQ.objects.create(u_id_id=u_id, b_id_id=b_id, s_id_id=s_id, t_id_id=topic_id, question=question, option_1=op1, option_2=op2, option_3=op3, option_4=op4, correct_ans=c_ans, explanation=exp, status= status, level=level)
        if mcq_details:
            return render(request, 'createMCQc.html', {'msg': "1"})
        else:
            return render(request, 'createMCQc.html', {'msg': "0"})
    else:
        subject_details = Subject.objects.filter(b_id_id=b_id).values()
        topic_details = Topic.objects.filter(b_id_id=b_id).values()
        return render(request, 'createMCQc.html', {"msg1": subject_details, "msg2": topic_details})

def generate_quest_ai_bool(request):
    if request.method == "POST":
        passage = request.POST['raw_data']
        truefalse = "yes"
        text = "truefalse: %s passage: %s </s>" % (passage, truefalse)
        max_len = 256
        encoding = tokenizer.encode_plus(text, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
        print("Context: ", passage)
        # print ("\nGenerated Question: ",truefalse)
        # # output = greedy_decoding(input_ids,attention_masks)
        # # print ("\nGreedy decoding:: ",output)
        output1 = beam_search_decoding(input_ids, attention_masks)
        print("\nBeam decoding [Most accurate questions] ::\n")
        for out1 in output1:
            print(out1)
        output2 = topkp_decoding(input_ids, attention_masks)
        print("\nTopKP decoding [Not very accurate but more variety in questions] ::\n")
        for out2 in output2:
            print(out2)
        return render(request, 'ViewBoolQs.html', {'msg': output1})
    else:
        return render(request, 'genQ.html')

@teacher_login_required
def read_mcqs(request):
    if request.method == 'GET':
        u_id = request.session['t_id']
        mcq_details = MCQ.objects.filter(u_id_id=u_id).values()
        return render(request, 'ViewMCQ.html', {"msg": mcq_details})

@contributor_login_required
def read_mcqs_contributor(request):
    if request.method == 'GET':
        u_id = request.session['c_id']
        mcq_details = MCQ.objects.filter(u_id_id=u_id).values()
        return render(request, 'ViewContribMCQ.html', {"msg": mcq_details})

@experts_login_required       
def approve_view_mcq(request):
    if request.method == 'GET':
        mcq_details = MCQ.objects.filter(status='2').values()
        return render(request, 'approveMCQ.html', {"msg": mcq_details})

@experts_login_required
def approve_mcqs(request, id, approve):  
    # b_id = request.session['b_id']
    mcq_details = MCQ.objects.filter(id=id).update(status=str(approve))
    if mcq_details:
        if str(approve) == '1':
            return render( request,'Approved.html', {"msg": "1"})
        elif str(approve) == '0':
            return render( request,'Rejected.html', {"msg": "0"})
    else:
        return render( request,'Rejected.html', {"msg": "3"})

def generate_mcq(request):
    if request.method == "POST":
        s_id = request.POST['s_id']
        topic_id = request.POST['topic_id']
        level = request.POST['level']
        status = '1'
        mcq_details = MCQ.objects.filter(s_id_id=s_id, t_id_id=topic_id, status=status, level=level).values()
        return render(request, 'generatedMCQ.html', {"msg": mcq_details})
    else:
        subject_details = Subject.objects.values()
        topic_details = Topic.objects.values()
        return render(request, 'generateMCQ.html', {"msg1": subject_details, "msg2": topic_details})

@experts_login_required
def contributor_delete(request):
    if request.method == "GET":
        board_users_details = Users.objects.filter(user_type="5").values()
        return render(request, 'ContributorReject.html', {"msg": board_users_details})

@experts_login_required
def contributor_delete_id(request,id):
    board_users_details = Users.objects.filter(id=id,user_type="5").delete()
    email_from = settings.EMAIL_HOST_USER
    recipient_list = ["2021.chinmay.vyapari@ves.ac.in"]
    send_mail("ACC BANNED", "Your acc is deleted", email_from, recipient_list)
    if board_users_details:
        return render(request, 'Deleted.html', {"msg": 1})
    else:
        return render(request, 'Deleted.html', {"msg": 0})

@experts_login_required
def contributorwise_mcq_display(request):
    if request.method == "POST":
        c_id = request.POST['c_id']
        board_users_details = MCQ.objects.filter(u_id_id=c_id).values()
        return render(request, 'MCQcontrib.html', {"msg": board_users_details})
    else:
        board_users_details = Users.objects.filter(user_type="5").values()
        return render(request, 'ContributorWiseData.html', {"msg": board_users_details})

@experts_login_required
def checkSimilarQuestions(request):
    if request.method == "POST":
        s_id = request.POST['s_id']
        topic_id = request.POST['topic_id']
        similar_filter = MCQ.objects.filter(s_id_id=s_id, t_id_id=topic_id).values()
        # temp_data = np.matrix()
        # for sm in similar_filter:
        #     pass
        #     temp_data.aee
        # df = pd.DataFrame(columns=["ID","DESCRIPTION"], data=np.matrix([[10,"Cancel ASN WMS Cancel ASN"],
        #                                                         [11,"MAXPREDO Validation is corect"],
        #                                                         [12,"Move to QC"],
        #                                                         [13,"Cancel ASN WMS Cancel ASN"],
        #                                                         [14,"MAXPREDO Validation is right"],
        #                                                         [15,"Verify files are sent every hours for this interface from Optima"],
        #                                                         [16,"MAXPREDO Validation are correct"],
        #                                                         [17,"Move to QC"],
        #                                                         [18,"Verify files are not sent"]
        #                                                         ]))
        # corpus = list(df["DESCRIPTION"].values)
        # vectorizer = TfidfVectorizer()
        # X = vectorizer.fit_transform(corpus)

        # threshold = 0.4

        # for x in range(0, X.shape[0]):
        #     for y in range(x, X.shape[0]):
        #         if(x != y):
        #             if(cosine_similarity(X[x], X[y]) > threshold):
        #                 print(df["ID"][x], ":", corpus[x])
        #                 print(df["ID"][y], ":", corpus[y])
        #                 print("Cosine similarity:", cosine_similarity(X[x], X[y]))
        #                 print()

        # if similar_filter
        return render(request, 'ViewSimilarQuestions.html', {"msg": similar_filter})

    else:
        subject_details = Subject.objects.values()
        topic_details = Topic.objects.values()
        return render(request, 'SelectSimilarQuestions.html', {"msg1": subject_details, "msg2": topic_details})
