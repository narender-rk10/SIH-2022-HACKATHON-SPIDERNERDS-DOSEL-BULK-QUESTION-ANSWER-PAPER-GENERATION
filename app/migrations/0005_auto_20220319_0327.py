# Generated by Django 3.0.8 on 2022-03-18 21:57

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0004_auto_20220319_0322'),
    ]

    operations = [
        migrations.AlterField(
            model_name='mcq',
            name='b_id',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='app.Board'),
        ),
    ]
