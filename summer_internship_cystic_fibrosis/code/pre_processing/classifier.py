from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def data_split(df, test_size):
    np_df = df.values

    train_set, test_set = train_test_split(np_df, test_size=test_size, random_state=42, stratify=np_df[:,-1])

    # Get the X and y for train, val and test
    X_train = train_set[:,:-1]
    y_train = train_set[:,-1]
    X_test = test_set[:,:-1]
    y_test = test_set[:,-1]
    
    # print(f'Shapes are {[X_train.shape,y_train.shape,X_test.shape,y_test.shape]}')
    
    return X_train,y_train,X_test,y_test

def svm(X_train,y_train,X_test,y_test,name,kernel,c):
    preproc_pl = Pipeline([ ('imputer', SimpleImputer(strategy="median")),
                        ('std_scaler', StandardScaler())])
    
    svm_pl = Pipeline([('preproc',preproc_pl),
                       ('svc',SVC(kernel=kernel, C=c, random_state=42))])
    svm_pl.fit(X_train,y_train)

    y_train_pred_svm = svm_pl.predict(X_train)
    y_test_pred_svm = svm_pl.predict(X_test)

    acc_train = accuracy_score(y_train,y_train_pred_svm)
    acc_test = accuracy_score(y_test,y_test_pred_svm)
    
    print('\033[1m' + name + '\033[0m')
    print()
    print(f'Training accuracy score = {acc_train}')
    print(f'Testing accuracy score = {acc_test}')


def random_forest(X_train, y_train, X_test, y_test, name, n_estimators):
    preproc_pl = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])

    rf_pl = Pipeline([
        ('preproc', preproc_pl),
        ('rf', RandomForestClassifier(n_estimators=n_estimators, random_state=42))
    ])
    
    rf_pl.fit(X_train, y_train)

    y_train_pred_rf = rf_pl.predict(X_train)
    y_test_pred_rf = rf_pl.predict(X_test)

    acc_train = accuracy_score(y_train, y_train_pred_rf)
    acc_test = accuracy_score(y_test, y_test_pred_rf)
    
    print('\033[1m' + name + '\033[0m')
    print()
    print(f'Training accuracy score = {acc_train}')
    print(f'Testing accuracy score = {acc_test}')

def decision_tree(X_train, y_train, X_test, y_test, name, max_depth=None):
    preproc_pl = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])

    dt_pl = Pipeline([
        ('preproc', preproc_pl),
        ('dt', DecisionTreeClassifier(max_depth=max_depth, random_state=42))
    ])
    
    dt_pl.fit(X_train, y_train)

    y_train_pred_dt = dt_pl.predict(X_train)
    y_test_pred_dt = dt_pl.predict(X_test)

    acc_train = accuracy_score(y_train, y_train_pred_dt)
    acc_test = accuracy_score(y_test, y_test_pred_dt)
    
    print('\033[1m' + name + '\033[0m')
    print()
    print(f'Training accuracy score = {acc_train}')
    print(f'Testing accuracy score = {acc_test}')


def xg_boost(X_train, y_train, X_test, y_test, name, n_estimators=100):
    preproc_pl = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
    ])

    xgb_pl = Pipeline([
        ('preproc', preproc_pl),
        ('xgb', XGBClassifier(n_estimators=n_estimators, random_state=42))
    ])
    
    xgb_pl.fit(X_train, y_train)

    y_train_pred_xgb = xgb_pl.predict(X_train)
    y_test_pred_xgb = xgb_pl.predict(X_test)

    acc_train = accuracy_score(y_train, y_train_pred_xgb)
    acc_test = accuracy_score(y_test, y_test_pred_xgb)
    
    print('\033[1m' + name + '\033[0m')
    print()
    print(f'Training accuracy score = {acc_train}')
    print(f'Testing accuracy score = {acc_test}')