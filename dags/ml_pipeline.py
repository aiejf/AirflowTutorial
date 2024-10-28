from airflow.models import DAG
from datetime import datetime
from airflow.operators.python import PythonOperator
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def prepare_data():
    import pandas as pd
    print("--- Inside prepare_data component ---")
    #Load dataset
    df=pd.read_csv("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")
    df=df.dropna()
    df.to_csv(f'final_df.csv', index=False)

def train_test_split():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    print("--- Inside train_test_split component ---")
    final_data=pd.read_csv(f'final_df.csv')
    target_column='species'
    X=final_data.loc[:,final_data.columns != target_column]
    y=final_data.loc[:,final_data.columns == target_column]

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,stratify=y,random_state=47)

    np.save(f'X_train.npy', X_train)
    np.save(f'X_test.npy', X_test)
    np.save(f'y_train.npy', y_train)
    np.save(f'y_test.npy', y_test)

    print("\n---- X_train ---")
    print("\n")
    print(X_train)

    print("\n---- X_test ---")
    print("\n")
    print(X_test)

    print("\n---- y_train ---")
    print("\n")
    print(y_train)

    print("\n---- y_test ---")
    print("\n")
    print(y_test)


def training_basic_classifier():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    print("--- Inside train_test_split component ---")
    X_train=np.load(f'X_train.npy', allow_pickle=True)
    y_train=np.load(f'y_train.npy', allow_pickle=True)

    classifier=LogisticRegression(max_iter=500)
    classifier.fit(X_train,y_train)
    import pickle
    with open(f'model.pkl','wb') as f:
        pickle.dump(classifier,f)

    print("\n--- logistic regression classifier is trained on iris data and saved to PV location /modle.pkl ---")


def predict_on_test_data():
    import numpy as np
    import pandas as pd
    import pickle
    print("--- Inside predict_on_test_data component ---")
    with open(f'model.pkl','rb') as f:
        logistic_reg_model=pickle.load(f)
    X_test=np.load(f'X_test.npy', allow_pickle=True)
    y_pred=logistic_reg_model.predict(X_test)
    np.save(f'y_pred.npy',y_pred)

    print("\n--- Predicted classes ---")
    print("\n")
    print(y_pred)


def predict_prob_on_test_data():
    import numpy as np
    import pandas as pd
    import pickle
    print("--- Inside predict_prob_on_test_data component ---")
    with open(f'model.pkl','rb') as f:
        logistic_reg_model=pickle.load(f)
    X_test=np.load(f'X_test.npy', allow_pickle=True)
    y_pred_prod=logistic_reg_model.predict_proba(X_test)
    np.save(f'y_pred_prod.npy',y_pred_prod)

    print("\n--- Predicted probabilities ---")
    print("\n")
    print(y_pred_prod)


def get_metrics():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score,log_loss
    from sklearn import metrics
    print("--- Inside get_metrics component ---")
    y_test=np.load(f'y_test.npy', allow_pickle=True)
    y_pred=np.load(f'y_pred.npy', allow_pickle=True)
    y_pred_prod=np.load(f'y_pred_prod.npy', allow_pickle=True)
    acc=accuracy_score(y_test,y_pred)
    pred=precision_score(y_test,y_pred,average='micro')
    recall=recall_score(y_test,y_pred,average='micro')
    entropy=log_loss(y_test,y_pred_prod)

    y_test=np.load(f'y_test.npy', allow_pickle=True)
    y_pred=np.load(f'y_pred.npy', allow_pickle=True)

    print(metrics.classification_report(y_test,y_pred))


with DAG(dag_id='ml_pipeline_demo',
         schedule='@daily',
         start_date=datetime(2024,10,26),
         catchup=False) as dag:

    task_prepare_data = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data,
    )

    task_train_test_split = PythonOperator(
        task_id='train_test_split',
        python_callable=train_test_split,
    )

    task_training_basic_classifier = PythonOperator(
        task_id='training_basic_classifier',
        python_callable=training_basic_classifier,
    )

    task_predict_on_test_data = PythonOperator(
        task_id='predict_on_test_data',
        python_callable=predict_on_test_data,
    )

    task_predict_prob_on_test_data = PythonOperator(
        task_id='predict_prob_on_test_data',
        python_callable=predict_prob_on_test_data,
    )

    task_get_metrics = PythonOperator(
        task_id='get_metrics',
        python_callable=get_metrics,
    )

    




    task_prepare_data>>    task_train_test_split >>    task_training_basic_classifier >>    task_predict_on_test_data >> task_predict_prob_on_test_data >>   task_get_metrics 
