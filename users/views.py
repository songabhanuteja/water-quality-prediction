from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def Viewdata(request):
    import os
    import pandas as pd
    from django.conf import settings
    path =os.path.join(settings.MEDIA_ROOT,'water_potability.csv')
    #path = os.path.join(settings.MEDIA_ROOT, "World_Happiness_2015_2017_.csv")
    data = pd.read_csv(path, nrows=500)
    print(data)
    data = data.to_html()
    return render (request, "users/Viewdata.html", {"data":data})
def preprocess(request):
    import os
    import pandas as pd
    from django.conf import settings
    path =os.path.join(settings.MEDIA_ROOT,'danaiahpreposedataset.csv')
    
    dani = pd.read_csv(path,nrows=500)
    print(dani)
    dani = dani.to_html()
    return render (request, "users/preprocessdata.html", {"dani":dani})

def ML(request):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from django.conf import settings
    import os
    path = os.path.join(settings.MEDIA_ROOT + "\\" + "danaiahpreposedataset.csv")
    df = pd.read_csv(path)
    #print(df.shape)
    #df.head()
    #df.info()
     #Seperating the data and labels
    X = df.drop('Potability',axis=1)
    Y= df['Potability']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=101,shuffle=True)
    Y_train.value_counts()
    Y_test.value_counts()

    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    dt=DecisionTreeClassifier(criterion= 'gini', min_samples_split= 100,max_depth=20,max_features='sqrt', splitter= 'best',max_leaf_nodes=10,random_state=42)
    dt.fit(X_train,Y_train)

    model = DecisionTreeClassifier()
    model.fit(X_train,Y_train)
    y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(Y_test, y_pred) * 100
    print('Accuracy:', accuracy)
    from sklearn.metrics import precision_score
    precision1 = precision_score(Y_test, y_pred) * 100
    print('Precision Score:', precision1)
    from sklearn.metrics import recall_score
    recall1 = recall_score(Y_test, y_pred) * 100
    print('recall_score:',recall1)
    from sklearn.metrics import f1_score
    f1score1 = f1_score(Y_test, y_pred) * 100
    print('f1score:',f1score1)
    # from sklearn.metrics import confusion_matrix
    # confusionmatrix = confusion_matrix(Y_test, y_pred) * 100
    # print('confusionmatrix:',confusionmatrix)



    model2 = KNeighborsClassifier()
    model2.fit(X_train, Y_train)
    y_pred2 = model2.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy2 = accuracy_score(Y_test, y_pred2) * 100
    print('Accuracy2:', accuracy2)
    from sklearn.metrics import precision_score
    precision2 = precision_score(Y_test, y_pred2) * 100
    print('precision2:',precision2)
    from sklearn.metrics import recall_score
    recall2 = recall_score(Y_test, y_pred2) * 100
    print('recall2:',recall2)  

    from sklearn.metrics import f1_score
    f1score2 = f1_score(Y_test, y_pred2) * 100
    print('f1score2:',f1score2)  
    # from sklearn.metrics import confusion_matrix
    # confusionmatrix2 = confusion_matrix(Y_test, y_pred2) * 100
    # print('confusionmatrix2:',confusionmatrix2)

   

    model3 = RandomForestClassifier()
    model3.fit(X_train, Y_train)
    y_pred3 = model3.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy3 = accuracy_score(Y_test, y_pred3) * 100
    print('Accuracy3:', accuracy3)
    from sklearn.metrics import precision_score
    precision3 = precision_score(Y_test, y_pred3) * 100
    print('precision3:',precision3)
    from sklearn.metrics import recall_score
    recall3 = recall_score(Y_test, y_pred3) * 100
    print('recall3:',recall3)    

    from sklearn.metrics import f1_score
    f1score3 = f1_score(Y_test, y_pred3) * 100
    print('f1score3:',f1score3)
    # from sklearn.metrics import confusion_matrix
    # confusionmatrix3 = confusion_matrix(Y_test, y_pred) * 100
    # print('confusionmatrix3:',confusionmatrix3)

    from sklearn.ensemble import GradientBoostingClassifier
    model4 = GradientBoostingClassifier()
    model4.fit(X_train, Y_train)
    y_pred4 = model4.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy4 = accuracy_score(Y_test, y_pred4) * 100
    print('Accuracy4:', accuracy4)
    from sklearn.metrics import precision_score
    precision4 = precision_score(Y_test, y_pred4) * 100
    print('precision4:',precision4)
    from sklearn.metrics import recall_score
    recall4 = recall_score(Y_test, y_pred4) * 100
    print('recall4:',recall4)    

    from sklearn.metrics import f1_score
    f1score4 = f1_score(Y_test, y_pred4) * 100
    print('f1score3:',f1score4)



    accuracy = {'DT': accuracy, 'KNN': accuracy2, 'RF': accuracy3,'GBC':accuracy4}
    precision = {'DT': precision1, 'KNN': precision2, 'RF': precision3,'GBC':precision4}
    recall = {'DT': recall1, 'KNN': recall2, 'RF': recall3,'GBC':recall4}
    f1score = {'DT':f1score1,'KNN':f1score2,'RF':f1score3,'GBC':f1score4}

    # confusionmatrix = {'DT':confusionmatrix, 'KNN':confusionmatrix2, 'RF':confusionmatrix3}
    #'confusionmatrix:',confusionmatrix
    # roc = {'RF': roc1, 'SVM': roc2, 'LogisticRegression': roc,  'MLP': roc5}
    return render(request, 'users/ML.html',
                  {"accuracy": accuracy, "precision": precision, "recall":recall, 'f1score':f1score})


   

def prediction(request):
    if request.method == 'POST':
        ph = int(request.POST.get('ph'))
        Hardness = int(request.POST.get('Hardness'))
        Solids= float(request.POST.get('Solids'))
        Chloramines = int(request.POST.get('Chloramines'))
        Sulfate	= int(request.POST.get('Sulfate'))
        Conductivity = int(request.POST.get('Conductivity'))
        Organic_carbon = request.POST.get('Organic_carbon')
        Trihalomethanes = request.POST.get('Trihalomethanes')
        Turbidity = request.POST.get('Turbidity')
        userinput = [[ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Trihalomethanes,Organic_carbon,Turbidity]]
        print(userinput)
        from .utility import prediction        
        test_pred = prediction.classification(userinput)
        print("Test Result is:", test_pred)
        if test_pred[0] == 1:
            msg = 'NOT QUALITY WATER'
        else:
            msg= 'QUALITY WATER'
            # return render(request, "users/predictions_form.html", {"test_data": test_pred, "result": rslt})
        return render(request, "users/predictions_form.html", {"msg":msg})
    else:
        return render(request, 'users/predictions_form.html', {})
    return render(request, 'users/predictions_form.html')
    
        
