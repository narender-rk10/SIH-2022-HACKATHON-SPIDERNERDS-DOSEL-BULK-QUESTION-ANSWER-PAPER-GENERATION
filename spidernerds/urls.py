"""spidernerds URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from app import views as av


urlpatterns = [
    path('', av.index),
    path('DoselLogin', av.dosel_login),
    path('BoardLogin', av.board_login),
    path('TeacherLogin', av.teacher_login),
    path('ExpertLogin', av.expert_login),
    path('ContributorLogin', av.contributor_login),
    path('logout', av.logout),
    path('dosel_index', av.dosel_index),
    path('board_index', av.board_index),
    path('teacher_index', av.teacher_index),
    path('contributor_index', av.contributor_index),
    path('experts_index', av.experts_index),
    path('add-board', av.add_board),
    path('read-board', av.read_boards),
    path('add-board-users',av.add_board_users),
    path('read-board-users',av.read_boards_users),
    path('add-sub', av.add_sub),
    path('read-sub', av.read_subjects),
    path('add-topic', av.add_topic),
    path('read-topic', av.read_topic),
    path('add-users',av.add_users),
    path('read-users',av.read_users),
    path('create-mcq', av.create_mcq),
    path('create-mcq-c', av.create_mcq_contributor),
    path('read-mcq', av.read_mcqs),
    path('read-mcq-c', av.read_mcqs_contributor),
    path('generate-ai-bool',av.generate_quest_ai_bool),
    path('approve-view-mcq', av.approve_view_mcq),
    path('approve-mcqs/<int:id>/<int:approve>', av.approve_mcqs),
    path('generate-mcq',av.generate_mcq),
    path('contributor-delete',av.contributor_delete),
    path('contributor-delete-id/<int:id>',av.contributor_delete_id),
    path('contributorwise-mcq-display',av.contributorwise_mcq_display),
    path('check-similar-questions',av.checkSimilarQuestions),
]
