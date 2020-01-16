from tkinter import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

df=pd.read_csv("hire_data1.csv")

win=Tk()

win.configure(bg="cyan")
fr1=Frame(win,width=1500,height=200,bg='white',bd=10,relief="raised")
fr1.pack(side=TOP)
lb=Label(fr1,text="HIRE PREDICTOR SYSTEM",fg="khaki",bg="black",font=('times new roman',40,'bold'))
lb.grid(row=0,padx=10,pady=10)
fr=Frame(win,width=5000,height=9000,bg='gray24',bd=10,relief="raised")

fr.pack(padx=15,pady=15)

win.geometry("9000x9000")

win.title("Hire Predictor System")
lb1=Label(fr,text="Name ",fg="red",bg='khaki2',font=('times new roman',18,'bold italic'))
lb1.grid(row=0,padx=10,pady=10,sticky=W)
lb2=Label(fr,text="Percentage ",bg="khaki2",fg="red",font=('times new roman',18,'bold italic'))
lb2.grid(row=2,padx=10,pady=10,sticky=W)

lb3=Label(fr,text="Backlogs ",bg="khaki2",fg="red",font=('times new roman',18,'bold italic'))
lb3.grid(row=4,padx=10,pady=10,sticky=W)

lb4=Label(fr,text="Internships ",bg="khaki2",fg="red",font=('times new roman',18,'bold italic'))
lb4.grid(row=6,padx=10,pady=10,sticky=W)
lb5=Label(fr,text="First Round",bg="khaki2",fg="red",font=('times new roman',18,'bold italic'))
lb5.grid(row=8,padx=10,pady=10,sticky=W)
lb7=Label(fr,text="Hired",bg="khaki2",fg="red",font=('times new roman',18,'bold italic'))
lb7.grid(row=12,padx=10,pady=10,sticky=W)

lb6=Label(fr,text=" Communication",bg="khaki2",fg="red",font=('times new roman',18,'bold italic'))
lb6.grid(row=10,padx=10,pady=10,sticky=W)

logreg=StringVar()
elo=Entry(fr,textvariable=logreg)
elo.grid(row=26,padx=10,pady=10)

dtree=StringVar()
edt=Entry(fr,textvariable=dtree)
edt.grid(row=26,column=2,padx=10,pady=10)

rf=StringVar()
erf=Entry(fr,textvariable=rf)
erf.grid(row=26,column=3,padx=10,pady=10)

svm1=StringVar()
esvm=Entry(fr,textvariable=svm1)
esvm.grid(row=26,column=4,padx=10,pady=10)

finalbtn=Button(fr,text="Final Submit",fg="white",bg="black",width=30,font=('times new roman',10,'bold'))
finalbtn.grid(row=30,column=2,padx=10,pady=15)


name=StringVar()
e1=Entry(fr,textvariable=name,)
e1.grid(row=0,column=2,padx=10,pady=10)

percent=StringVar()
e2=Entry(fr,textvariable=percent)
e2.grid(row=2,column=2,padx=10,pady=10)

back=StringVar()
e3=Entry(fr,textvariable=back)
e3.grid(row=4,column=2,padx=10,pady=10)

intr=StringVar()
e4=Entry(fr,textvariable=intr)
e4.grid(row=6,column=2,padx=10,pady=10)

firstr=StringVar()
e5=Entry(fr,textvariable=firstr)
e5.grid(row=8,column=2,padx=10,pady=10)


comsk=StringVar()
e6=Entry(fr,textvariable=comsk,)
e6.grid(row=10,column=2,padx=10,pady=10)

hired=StringVar()
e7=Entry(fr,textvariable=hired)
e7.grid(row=12,column=2,padx=10,pady=10)

#defining Functins for different algorithms

#logistic Regression
def lr():
    
    X = df.iloc[:,:5].values
    y = df.iloc[:,5].values
    model=LogisticRegression()
    model.fit(X,y)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
    n = e1.get()
    p=e2.get()
    b=e3.get()
    i=e4.get()
    f=e5.get()
    c=e6.get()

    print(f" Logistic regression : the name is {n} \n percentage is {p} \n number of backlogs is {b} \n number of internships done is {i} \n marks in first round is {f} \n marks in communication round is {c}")

    
    candi = np.array([p,b,i,f,c], dtype =float).reshape(1,-1)
    a=model.predict(candi)

    print(a)
    if a==1:
        elo.insert(0,"hired")
    else:
        elo.insert(0,"not hired")

#Decision Tree
def dt():
    
    X = df.iloc[:,:5].values
    y = df.iloc[:,5].values
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 100)
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
    clf_gini.fit(X_train, y_train)
    n = e1.get()
    p=e2.get()
    b=e3.get()
    i=e4.get()
    f=e5.get()
    c=e6.get()
    print(f" Decision Tree: the name is {n} \n percentage is {p} \n number of backlogs is {b} \n number of internships done is {i} \n marks in first round is {f} \n marks in communication round is {c}")

    a=clf_gini.predict([[p,b,i,f,c]])
    print(a)
    if a==1:
        edt.insert(0,"hired")
    else:
        edt.insert(0,"not hired")


#random Forest
def rf():
    
    X = df.iloc[:,:5].values
    y = df.iloc[:,5].values
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X, y)
    n = e1.get()
    p=e2.get()
    b=e3.get()
    i=e4.get()
    f=e5.get()
    c=e6.get()
    print(f"Random Forest: the name is {n} \n percentage is {p} \n number of backlogs is {b} \n number of internships done is {i} \n marks in first round is {f} \n marks in communication round is {c}")


    a=clf.predict([[p,b,i,f,c]])
    print(a)
    if a==1:
        erf.insert(0,"hired")
    else:
        erf.insert(0,"not hired")

#support vector Machine
def sv():
    
    X = df.iloc[:,:5].values
    y = df.iloc[:,5].values

    svcmodel = svm.SVC(kernel='linear', C=1)
    svcmodel.fit(X, y)
    n = e1.get()
    p=e2.get()
    b=e3.get()
    i=e4.get()
    f=e5.get()
    c=e6.get()
    a=svcmodel.predict(np.array([p,b,i,f,c],dtype='float').reshape(1,-1))
    
    print(f" Support Vector Machine: the name is {n} \n percentage is {p} \n number of backlogs is {b} \n number of internships done is {i} \n marks in first round is {f} \n marks in communication round is {c}")

    if a==1:
        esvm.insert(0,"hired")
    else:
        esvm.insert(0,"not hired")

#defing the Submit button
def submit():
    
    rs1=elo.get()
    rs2=edt.get()
    rs3=erf.get()
    rs4=esvm.get()
    results=[rs1,rs2,rs3,rs4]
    if results.count('hired')>results.count('not hired'):
       hired.set('hired')
       h=1
    else:
        hired.set('not hired')
        h=0

logregbtn=Button(fr,text="Logistic Regression",command=lr,fg="yellow",bg="red",font=('times new roman',11,'bold'))
logregbtn.grid(row=2,column=4,padx=10,pady=10,sticky=W)

dectreebtn=Button(fr,text=" Decision Tree",command=dt,bg="red",fg="yellow",font=('times new roman',11,'bold'))
dectreebtn.grid(row=4,column=4,padx=10,pady=10)

raforestbtn=Button(fr,text="Random Forest ",bg="red",command=rf,fg="yellow",font=('times new roman',11,'bold'))
raforestbtn.grid(row=6,column=4,padx=10,pady=10)

svmbtn=Button(fr,text="SVM",bg="red",command=sv,fg="yellow",font=('times new roman',11,'bold'))
svmbtn.grid(row=8,column=4,padx=10,pady=10)


lblog=Label(fr,text="Logistic regression",fg="white",width=16,bg='black',font=('times new roman',13,'bold italic'))
lblog.grid(row=14,padx=10,pady=10)
lbdtree=Label(fr,text="Decision Tree ",fg="white",width=15,bg="black",font=('times new roman',13,'bold italic'))
lbdtree.grid(row=14,column=2,padx=10,pady=10)

lbrforest=Label(fr,text="Random forest ",width=15,bg="black",fg="white",font=('times new roman',13,'bold italic'))
lbrforest.grid(row=14,column=3,padx=10,pady=10)

lbsvm=Label(fr,text="SVM ",bg="black",width=13,fg="white",font=('times new roman',13,'bold italic'))
lbsvm.grid(row=14,column=4,padx=10,pady=10,sticky=W)



logreg=StringVar()
elo=Entry(fr,textvariable=logreg)
elo.grid(row=26,padx=10,pady=10)

dtree=StringVar()
edt=Entry(fr,textvariable=dtree)
edt.grid(row=26,column=2,padx=10,pady=10)

rf=StringVar()
erf=Entry(fr,textvariable=rf)
erf.grid(row=26,column=3,padx=10,pady=10)

svm1=StringVar()
esvm=Entry(fr,textvariable=svm1)
esvm.grid(row=26,column=4,padx=10,pady=10)

finalbtn=Button(fr,text="Final Submit",command=submit,fg="white",bg="black",width=30,font=('times new roman',10,'bold'))
finalbtn.grid(row=30,column=2,padx=10,pady=15)
win.mainloop()
