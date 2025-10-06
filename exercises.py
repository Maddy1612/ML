Program 1a:

import numpy as np
def initialize_student_data(num_students):
    student_data=[]
    for i in range(num_students):
        name=input(f"Get student's name for student {i+1}:")
        age=int(input(f"Get age for age for {name}:"))
        math_score=float(input(f"Get math score for {name}:"))
        science_score=float(input(f"Get science score for {name}:"))
        physics_score=float(input(f"Get physics score for {name}:"))
        chemistry_score=float(input(f"Get chemistry score for {name}:"))
        student_data.append([name,age,math_score,science_score,physics_score,chemistry_score])
    student_data=np.array(student_data)
    return student_data
def calculate_overall_average(student_data):
    scores=student_data[:,2:].astype(float)
    overall_avg=np.mean(scores)
    return overall_avg
def top_students_overall(student_data,n):
    scores=student_data[:,2:].astype(float)
    overall_avg_scores=np.mean(scores,axis=1)
    top_indices=np.argsort(overall_avg_scores)[::-1][:n]
    top_students=student_data[top_indices]
    return top_students
def filter_students(student_data,min_age,min_score,subject='Math'):
    subject_index={'Math':2,'Science':3,'Physics':4,'Chemistry':5}[subject]
    filtered_students=student_data[(student_data[:,1].astype(int)>=min_age) & (student_data[:,subject_index].astype(float)>=min_score)]
    return filtered_students
num_students=int(input("Get the number of students:"))
student_data=initialize_student_data(num_students)
print("\nInitial Student Data:")
print(student_data)
print()
overall_avg=calculate_overall_average(student_data)
print(f"Overall Average Score for Students: {overall_avg:.2f}")
print()
top_n=int(input("Get the number of top students to display:"))
top_students=top_students_overall(student_data,top_n)
print(f"\nTop {top_n} Students based on Overall Average Score:")
print(top_students)
print()
min_age_filter=int(input("Get the minimum age to filter students:"))
min_score_filter=float(input("Get the minimum score in Physics to filter students:"))
filtered_students_phy=filter_students(student_data,min_age_filter,min_score_filter,subject="Physics")
print(f"\nStudents aged {min_age_filter} or older with atleast {min_score_filter} in Physics:")
print(filtered_students_phy)
min_score_filter_chem=float(input("Get the minimum score in Chemistry to filter students:"))
filtered_students_chem=filter_students(student_data,min_age_filter,min_score_filter_chem,subject="Chemistry")
print(f"\nStudents aged {min_age_filter} or older with atleast {min_score_filter_chem} in Chemistry:")
print(filtered_students_chem)
--------------------------------------------------------------------------------------------------------------------------------------------------------

Program 1b:

from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
iris=load_iris()
x=iris.data
y=iris.target
df=pd.DataFrame(data=x,columns=iris.feature_names)
df['target']=y
missing_values=df.isnull().sum()
print("Missing Values =",missing_values)
summary_stats=df.describe()
print(summary_stats)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
clf=DecisionTreeClassifier(random_state=42)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy}")
conf_matrix=confusion_matrix(y_test,y_pred)
print(f"Confusion Matrix:{conf_matrix}")
-------------------------------------------------------------------------------------------------------------------------------------------------

Program 2a:

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
data=pd.DataFrame(cancer.data,columns=cancer.feature_names)
data['target']=cancer.target
print(data.info())
print(data.describe())
print(data.isnull().sum())
plt.figure(figsize=(10,6))
plt.plot(data.index,data['mean radius'],label='Mean Radius')
plt.title('Line Plot of Mean Radius')
plt.xlabel('Index')
plt.ylabel('Mean Radius')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10,6))
plt.scatter(data['mean radius'],data['mean texture'],c=data['target'],cmap='coolwarm',alpha=0.5)
plt.title('Scatter Plot of Mean Radius vs Mean Texture')
plt.xlabel('Mean Radius')
plt.ylabel('Mean Texture')
plt.grid(True)
plt.show()
plt.figure(figsize=(10,6))
plt.bar(data['target'].value_counts().index,data['target'].value_counts().values)
plt.title('Bar Plot of Target Class Distribution')
plt.xlabel('Target Class')
plt.ylabel('Count')
plt.xticks([0, 1],['Malignant','Benign'])
plt.grid(True)
plt.show()
plt.figure(figsize=(10,6))
plt.hist(data['mean area'],bins=30,alpha=0.7)
plt.title('Histogram of Mean Area')
plt.xlabel('Frequency')
plt.grid(True)
plt.show()
plt.figure(figsize=(10,6))
plt.boxplot([data[data['target']==0]['mean radius'],data[data['target']==1]['mean radius']],labels=['Malignant','Benign'])
plt.title('Box Plot of Mean Radius by Target Class')
plt.xlabel('Target Class')
plt.ylabel('Mean Radius')
plt.grid(True)
plt.show()
---------------------------------------------------------------------------------------------------------------------------------------------------

Program 2b:

import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer 
cancer = load_breast_cancer() 
data = pd.DataFrame(cancer.data, columns=cancer.feature_names) 
data['target'] = cancer.target 
print(data.info())
print(data.head(1))
print(data.describe()) 
print(data.isnull().sum()) 
plt.figure(figsize=(6, 4)) 
sns.countplot(x='target', data=data, palette='coolwarm') 
plt.title('Count Plot of Target Classes') 
plt.xlabel('Target Class') 
plt.ylabel('Count') 
plt.xticks([0, 1], ['Malignant', 'Benign']) 
plt.show()
plt.figure(figsize=(10, 6)) 
sns.kdeplot(data=data[data['target'] == 0]['mean radius'], shade=True, label='Malignant', color='r') 
sns.kdeplot(data=data[data['target'] == 1]['mean radius'], shade=True, label='Benign', color='b')
plt.title('KDE Plot of Mean Radius') 
plt.xlabel('Mean Radius') 
plt.ylabel('Density')
plt.legend() 
plt.show()
plt.figure(figsize=(10, 6))
sns.violinplot(x='target', y='mean radius', data=data, palette='coolwarm') 
plt.title('Violin Plot of Mean Radius by Target Class')
plt.xlabel('Target Class') 
plt.ylabel('Mean Radius')
plt.xticks([0, 1], ['Malignant', 'Benign']) 
plt.show() 
sns.pairplot(data, vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area'], hue='target', palette='coolwarm')
plt.title('Pair Plot') 
plt.show()
plt.figure(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap') 
plt.show()
--------------------------------------------------------------------------------------------------------------------------------------------

Program 3:

import pandas as pd
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix, accuracy_score
iris = load_iris() 
X = iris.data 
y = iris.target 
feature_names = iris.feature_names
print("Range of values before scaling:") 
for i, feature_name in enumerate(feature_names):
    print(f"{feature_name}: [{X[:, i].min()} to {X[:, i].max()}]")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nRange of values after scaling:") 
for i, feature_name in enumerate(feature_names):
    print(f"{feature_name}: [{X_train_scaled[:, i].min()} to {X_train_scaled[:, i].max()}]") 
print("\nOriginal Training Data (first 3 rows):") 
print(pd.DataFrame(X_train, columns=feature_names).head(3))
lda = LDA(n_components=2) 
X_train_lda = lda.fit_transform(X_train_scaled, y_train) 
X_test_lda = lda.transform(X_test_scaled)
print("\nTraining Data After LDA (first 3 rows):") 
print(pd.DataFrame(X_train_lda, columns=['LDA Component 1', 'LDA Component 2']).head(3)) 
print("\nExplained variance ratio:", lda.explained_variance_ratio_)
print("\nDimensions of the original dataset:", X_train.shape)
print("\nDimensions of the dataset after LDA:", X_train_lda.shape) 
knn_original = KNeighborsClassifier(n_neighbors=3)
knn_original.fit(X_train_scaled, y_train) 
y_pred_original = knn_original.predict(X_test_scaled) 
accuracy_original = accuracy_score(y_test, y_pred_original) 
conf_matrix_original = confusion_matrix(y_test, y_pred_original) 
print("\nKNN Classifier on Original 4D Features:")
print(f"Accuracy: {accuracy_original:.2f}") 
print("Confusion Matrix:\n", conf_matrix_original) 
knn_lda = KNeighborsClassifier(n_neighbors=3)
knn_lda.fit(X_train_lda, y_train)
y_pred_lda = knn_lda.predict(X_test_lda) 
accuracy_lda = accuracy_score(y_test, y_pred_lda) 
conf_matrix_lda = confusion_matrix(y_test, y_pred_lda) 
print("\nKNN Classifier on 2D LDA Features:")
print(f"Accuracy: {accuracy_lda:.2f}") 
print("Confusion Matrix:\n", conf_matrix_lda) 
----------------------------------------------------------------------------------------------------------------------------

Program 4:

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
iris=load_iris()
x=iris.data
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
pca=PCA(n_components=2)
x_pca=pca.fit_transform(x_scaled)
print("Principal Component Details")
print("\n Explained Variance Ratio:",pca.explained_variance_ratio_)
print("\n Principal Components:")
print(pca.components_)
k_means=KMeans(n_clusters=3,random_state=42,n_init=10)
k_means.fit(x_pca)
y_kmeans=k_means.predict(x_pca)
print("\n Cluster Centers(in PCA-reduced speed):")
for i, center in enumerate(k_means.cluster_centers_):
    print(f"Cluster {i+1}:{center}")
print("\n Cluster Sizes:")
for i, size in enumerate(np.bincount(y_kmeans)):
    print(f"Cluster {i+1}:{size}")
------------------------------------------------------------------------------------------------------------------

Program 5:

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.svm import SVC
dat=load_wine()
data=pd.DataFrame(dat.data,columns=dat.feature_names)
data['quality']=dat.target
print(data.info())
x=data.drop('quality',axis=1)
y=data['quality']
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.3,random_state=42)
model=SVC(kernel='rbf',class_weight='balanced',probability=True)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\n Confusion Matrix")
cm=confusion_matrix(y_test,y_pred)
print(cm)
precision=precision_score(y_test,y_pred,average=None)
recall=recall_score(y_test,y_pred,average=None)
f1=f1_score(y_test,y_pred,average=None)
print("\n Precision for each class:")
for i, p in enumerate(precision):
    print(f"Class {i}: {p:.4f}")
print("\n Recall for each class:")
for i, r in enumerate(recall):
    print(f"Class {i}: {r:.4f}")
print("\n F1 Score for each class:")
for i, f in enumerate(f1):
    print(f"Class {i}: {p:.4f}")
------------------------------------------------------------------------------------------------------------------------------------------

Program 6:

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve,roc_auc_score
import matplotlib.pyplot as plt
cancer=load_breast_cancer()
x=pd.DataFrame(cancer.data,columns=cancer.feature_names)
y=cancer.target
print("First few rows of Breast Cancer Dataset:")
print(x.head())
print("\n Target Variable Distribution:")
print(pd.Series(y).value_counts())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
model=LogisticRegression(max_iter=10000)
model.fit(x_train_scaled,y_train)
y_pred=model.predict(x_test_scaled)
y_prob=model.predict_proba(x_test_scaled)[:,1]
print("\n Model Evaluation:")
print("\n Accuracy:", accuracy_score(y_test,y_pred))
print("\n Confusion Matrix:", confusion_matrix(y_test,y_pred))
print("\n Classification Report:", classification_report(y_test,y_pred))
fpr,tpr,thresholds=roc_curve(y_test,y_prob)
auc=roc_auc_score(y_test,y_prob)
plt.figure()
plt.plot(fpr,tpr,color='darkorange',lw=2,label=f'ROC curve(area={auc:.2f})')
plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
----------------------------------------------------------------------------------------------------------------------------------

Program 7:

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
data=load_breast_cancer()
x=data.data
y=data.target
print("Feature Names:", data.feature_names)
print("Class Names:", data.target_names)
print("First two rows of the datasets:")
print(x[:2])
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
plt.figure(figsize=(20,10))
tree.plot_tree(clf,feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()
------------------------------------------------------------------------------------------------------------------------------------------

Program 8:

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.metrics import completeness_score, silhouette_score, calinski_harabasz_score
wine=load_wine()
x=pd.DataFrame(wine.data, columns=wine.feature_names)
y=wine.target
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
k=3
kmeans=KMeans(n_clusters=k, n_init=10, random_state=42)
kmeans.fit(x_scaled)
centroids=kmeans.cluster_centers_
labels=kmeans.labels_
completeness=completeness_score(y, labels)
silhouette_avg=silhouette_score(x_scaled, labels)
calinski_harabasz=calinski_harabasz_score(x_scaled, labels)
print(f"Silhouette Coefficient: {silhouette_avg:.2f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz:.2f}")
print(f"Completeness: {completeness:.2f}")
----------------------------------------------------------------------------------------------------------------------------------------------

Program 9:

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import completeness_score, silhouette_score, calinski_harabasz_score
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
iris = load_iris()
x = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
data = pd.concat([x, pd.Series(y, name='species')], axis=1)
sample = data.groupby('species').apply(lambda x: x.sample(10, random_state=42)).reset_index(drop=True)
x_sample = sample.drop(columns='species')
y_sample = sample['species']
scaler = StandardScaler()
x_sample_scaled = scaler.fit_transform(x_sample)
linked = sch.linkage(x_sample_scaled, method='ward')
num_clusters = 3
labels = sch.fcluster(linked, num_clusters, criterion='maxclust')
completeness = completeness_score(y_sample, labels)
silhouette = silhouette_score(x_sample_scaled, labels)
calinski_harabasz = calinski_harabasz_score(x_sample_scaled, labels)
print(f"Number of Clusters: {num_clusters}")
print(f"Completeness Score: {completeness:.3f}")
print(f"Silhouette Score: {silhouette:.3f}")
print(f"Calinski-Harabasz Score: {calinski_harabasz:.3f}")
plt.figure(figsize=(12, 8))
dendrogram = sch.dendrogram(
    linked,
    orientation='top',
    labels=y_sample.values,
    distance_sort='descending',
    show_leaf_counts=True
)
plt.title("Dendrogram of Hierarchical Clustering on Iris Sample")
plt.xlabel("Sample Index / Species")
plt.ylabel("Euclidean Distance")
plt.show()
-------------------------------------------------------------------------------------------------------------------------------------------
