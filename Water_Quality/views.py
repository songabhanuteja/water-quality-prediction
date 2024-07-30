from django.shortcuts import render
from users.forms import UserRegistrationForm

# Create your views here.
def home(request):
    return render(request,'index.html',{})

def AdminLogin(request):
    return render(request, 'AdminLogin.html', {})

def UserLogin(request):
    return render(request, 'UserLogin.html',{})


def UserRegister(request):
    form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def logout(request):
    return render(request,'index.html',{})