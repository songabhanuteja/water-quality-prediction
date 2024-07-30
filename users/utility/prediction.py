def classification(userinput):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    #sns.sns.set_style('darkgrid')
    from sklearn.model_selection import train_test_split
    #from sklearn.preprocessing import StandardScaler
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
    #z =df.drop('Unnamed')
    Y= df['Potability']
    print(X)
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=101,shuffle=True)
    Y_train.value_counts()
    Y_test.value_counts()

    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

    #using the DecisionTreeClassifier
    # dt=DecisionTreeClassifier(criterion= 'gini', min_samples_split= 100,max_depth=20,max_features='sqrt', splitter= 'best',max_leaf_nodes=10,random_state=42)
    # dt.fit(X_train,Y_train)

    # model = DecisionTreeClassifier()
    # model.fit(X_train,Y_train)
    # y_pred = model.predict(userinput)
    # print(y_pred)

    # return y_pred

    # dt=RandomForestClassifier(criterion= 'gini', min_samples_split= 100,max_depth=20,max_features='sqrt', splitter= 'best',max_leaf_nodes=10,random_state=42)
    rf=RandomForestClassifier()

    rf.fit(X_train,Y_train)

    model = DecisionTreeClassifier()
    model.fit(X_train,Y_train)
    y_pred = model.predict(userinput)
    print(y_pred)

    return y_pred




#hyperperameter turning

# import numpy as np
# from sklearn.model_selection import RandomizedSearchCV
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}


# from sklearn.ensemble import RandomForestClassifier
# rf= RandomForestClassifier()
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 19, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# # Fit the random search model
# rf_random.fit(X_train,Y_train)



# prediction=rf_random.predict(X_test)
# accuracy_rf=accuracy_score(Y_test,prediction)*100
# accuracy_rf



