from django.db import models

# Create your models here.

# DoSEL 0
# board 1
# institute 2
# teacher 3
# experts 4
# contributor 5

class DOSEL(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    password = models.CharField(max_length=32)
    contact = models.CharField(max_length=100)

    class Meta:
        db_table = "DOSEL"

class Board(models.Model):
    name = models.CharField(max_length=100)
    
    class Meta:
        db_table = "board"

class BoardUsers(models.Model):
    b_id = models.ForeignKey(Board, on_delete=models.CASCADE)
    email = models.EmailField()
    password = models.CharField(max_length=32)
    contact = models.CharField(max_length=100)
    
    class Meta:
        db_table = "boardusers"

class Users(models.Model):
    b_id = models.ForeignKey(Board, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    email = models.EmailField()
    password = models.CharField(max_length=32)
    contact = models.CharField(max_length=100)
    user_type = models.CharField(max_length=100)

    class Meta:
        db_table = "users"

class Subject(models.Model):
    b_id = models.ForeignKey(Board, on_delete=models.CASCADE)
    s_name = models.CharField(max_length=100)

    class Meta:
        db_table = "subject"


class Topic(models.Model):
    b_id = models.ForeignKey(Board, on_delete=models.CASCADE)
    s_id = models.ForeignKey(Subject, on_delete=models.CASCADE)
    t_name = models.CharField(max_length=100)

    class Meta:
        db_table = "topic"


class MCQ(models.Model):
    u_id = models.ForeignKey(Users, on_delete=models.CASCADE)
    b_id = models.ForeignKey(Board, on_delete=models.CASCADE)
    s_id = models.ForeignKey(Subject, on_delete=models.CASCADE)
    t_id = models.ForeignKey(Topic, on_delete=models.CASCADE)
    question = models.CharField(max_length=250)
    option_1 = models.CharField(max_length=100)
    option_2 = models.CharField(max_length=100)
    option_3 = models.CharField(max_length=100)
    option_4 = models.CharField(max_length=100)
    correct_ans = models.CharField(max_length=100)
    explanation = models.CharField(max_length=250)
    #0-rejected, 1-accepted, 2-pending
    status = models.CharField(max_length=100)
    #0-easy, 1-medium, 2-hard
    level = models.CharField(max_length=100)

    class Meta:
        db_table = "mcq"

# class BoardLinker(models.Model):
#     u_id = models.ForeignKey(Users, on_delete=models.CASCADE)
#     email = models.ForeignKey(Users, on_delete=models.CASCADE)
#     b_name = models.ForeignKey(Users, on_delete=models.CASCADE)
#     b_id

#     def __str__(self):
#         return f'{self.o_email} {self.o_password}'

#     class Meta:
#         db_table = "BoardLinker"
