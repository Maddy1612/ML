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
[8/28, 12:10] ᴍᴀᴅᴅʏ: Program 7: Solving Classification problem using Decision tree
Problem Statement:
The goal of this program is to build and evaluate a Decision Tree classifier using the Breast 
Cancer dataset to predict whether a tumor is benign or malignant. The program will load the 
dataset, explore its features, train a Decision Tree model, evaluate its performance, and 
visualize the resulting decision tree.
from sklearn.datasets import load_breast_cancer # Import function to load the Breast Cancer 
dataset
from sklearn.model_selection import train_test_split # Import function to split data into 
training and testing sets
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree classifier
from sklearn.metrics import accuracy_score, classification_report # Import functions to 
evaluate model performance
from sklearn import tree # Import module for visualizing decision trees
import matplotlib.pyplot as plt # Import plotting library
# Load Breast Cancer dataset
data = load_breast_cancer() # Load dataset and store in 'data' variable
X = data.data # Extract feature data (e.g., measurements) into 'X'
y = data.target # Extract target labels (e.g., benign or malignant) into 'y'
print("Feature names:", data.feature_names) # Print names of features
print("Class names:", data.target_names) # Print names of target classes
print("First two rows of the dataset:") # Print message indicating the start of dataset preview
print(X[:2]) # Print the first two rows of feature data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 
Split data into 70% training and 30% testing; set seed for reproducibility
clf = DecisionTreeClassifier() # Create an instance of the Decision Tree classifier
clf.fit(X_train, y_train) # Train the classifier on the training data
y_pred = clf.predict(X_test) # Predict labels for the test data
accuracy = accuracy_score(y_test, y_pred) # Compute the accuracy of the predictions
print("Accuracy:", accuracy) # Print the accuracy score
print("Classification Report:") # Print message indicating the start of classification report
print(classification_report(y_test, y_pred, target_names=data.target_names)) # Print detailed 
performance metrics
plt.figure(figsize=(20,10)) # Create a new figure with specified size (20x10 inches)
tree.plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, 
filled=True) # Plot the decision tree with feature names and class names
plt.show() # Display the plot
[8/28, 12:12] ᴍᴀᴅᴅʏ: Program 8: Supervised data compression via linear discriminant analysisProblem StatementThe goal is to analyze and visualize the Iris dataset using Linear Discriminant Analysis (LDA) to simplify the data and see how well different Iris flower species are separated. This involves building a machine learning pipeline that first normalizes the data to ensure consistency across features, applies LDA to reduce the number of features while maintaining the differences between flower types, and uses a classifier to categorize the Iris flowers based on the transformed data. The outcome includes evaluating the model’s performance and generating a scatter plot that illustrates the distribution of Iris species in the reduced feature spaceimport seaborn as sns # Importing seaborn for enhanced plottingimport pandas as pd # Importing pandas for DataFrame manipulationfrom sklearn.datasets import load_iris # Importing function to load Iris datasetfrom sklearn.discriminant_analysis import LinearDiscriminantAnalysis # Importing LDAfrom sklearn.model_selection import train_test_split # Importing function to split datafrom sklearn.pipeline import Pipeline # Importing Pipeline for creating workflowsfrom sklearn.preprocessing import StandardScaler # Importing scaler for normalizationfrom sklearn.neighbors import KNeighborsClassifier # Importing KNN classifierfrom sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Importing metrics for evaluationimport matplotlib.pyplot as plt # Importing matplotlib for plotting# Load the Iris datasetiris = load_iris() # Load the Iris dataset from sklearnX = iris.data # Feature matrix (sepal length, sepal width, petal length, petal width)y = iris.target # Target vector (species labels)class_names = iris.target_names # Names of the classes (species names)# Split the data into training and test setsX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # Split data into 70% train and 30% test# Determine the number of LDA componentsn_classes = len(np.unique(y)) # Number of unique classes (3 for Iris dataset)n_features = X.shape[1] # Number of features (4 for Iris dataset)n_components = min(n_classes - 1, n_features) # Number of LDA components (min of 2 or 4)# Create a pipeline with StandardScaler, LDA with dynamic components, and KNeighborsClassifierpipeline = Pipeline([('scaler', StandardScaler()), # Normalize the features to have zero mean and unit variance('lda', LinearDiscriminantAnalysis(n_components=n_components)), # Apply LDA with dynamic number of components('clf', KNeighborsClassifier()) # KNN classifier for classification])# Train the pipelinepipeline.fit(X_train, y_train) # Fit the pipeline on the training data# Extract the LDA component from the pipelinelda = pipeline.named_steps['lda'] # Access the LDA component from the pipeline for transformation# Transform the data with LDAX_train_lda = lda.transform(X_train) # Transform training data to LDA spaceX_test_lda = lda.transform(X_test) # Transform test data to LDA space# Predict on the test sety_pred = pipeline.predict(X_test) # Predict the target values for the test set# Evaluate the modelaccuracy = accuracy_score(y_test, y_pred) # Calculate the accuracy of the model on the test setprint(f"\nAccuracy: {accuracy:.2f}") # Print the accuracy# Classification reportreport = classification_report(y_test, y_pred, target_names=class_names) # Generate classification reportprint("\nClassification Report:") # Print header for classification reportprint(report) # Print classification report# Confusion matrixconf_matrix = confusion_matrix(y_test, y_pred) # Compute confusion matrix for the test setprint("Confusion Matrix: \n", conf_matrix) # Print confusion matrix# Prepare DataFrame for Seabornlda_df = pd.DataFrame(X_train_lda, columns=[f'LDA Component {i+1}' for i in range(n_components)]) # Create DataFrame for LDA componentslda_df['Class'] = y_train # Add class labels to DataFramelda_df['Class'] = lda_df['Class'].map({i: class_names[i] for i in range(len(class_names))}) # Map class integers to class names# Plot LDA components using Seabornplt.figure(figsize=(12, 6)) # Create a new figure with specified sizecomponent_x = 0 # Index of the x-axis component (first LDA component)component_y = 1 # Index of the y-axis component (second LDA component)# Create the scatter plot with Seabornsns.scatterplot(x=f'LDA Component {component_x + 1}', # X-axis labely=f'LDA Component {component_y + 1}', # Y-axis labelhue='Class', # Color points by class labelpalette='Paired', # Color palette for classesdata=lda_df, # DataFrame with LDA components and class labelsedgecolor='k' # Add black edge color to points for better visibility)# Add title and labelsplt.xlabel(f'LDA Component {component_x + 1}') # X-axis labelplt.ylabel(f'LDA Component {component_y + 1}') # Y-axis labelplt.title(f'LDA Components {component_x + 1} vs {component_y + 1}') # Plot title# Add legend inside the figure with adjusted positionplt.legend(title='Class Label', loc='upper right', bbox_to_anchor=(1, 1), frameon=True) # Legend plt.show() # Display the plot
[8/28, 12:13] ᴍᴀᴅᴅʏ: Program 9: Grouping objects by similarity using k-meansProblem Statement:The goal of this analysis is to use K-Means clustering on the Wine dataset and evaluate how well it identifies different wine types. The process includes standardizing the features of the data and applying K-Means clustering with k = 3, which matches the number of wine types. The quality of the clustering is then measured using three metrics: Completeness Score (to see how well data points of the same wine type are grouped together), Silhouette Coefficient (to assess how distinct and well-separated the clusters are), and Calinski-Harabasz Index (to evaluate the separation and compactness of the clusters). These metrics help determine if k = 3 is an effective choice for clustering this dataset.import pandas as pdfrom sklearn.cluster import KMeansfrom sklearn.preprocessing import StandardScalerfrom sklearn.datasets import load_winefrom sklearn.metrics import completeness_score, silhouette_score, calinski_harabasz_score # Load the Wine datasetwine = load_wine()# Extract features and targetX = pd.DataFrame(wine.data, columns=wine.feature_names)y = wine.target# Standardize the featuresscaler = StandardScaler()X_scaled = scaler.fit_transform(X)# Set the number of clustersk = 3# Initialize KMeans with n_init explicitly setkmeans = KMeans(n_clusters=k, n_init=10, random_state=42)# Fit the modelkmeans.fit(X_scaled)# Get cluster centers and labelscentroids = kmeans.cluster_centers_labels = kmeans.labels_# Calculate evaluation metricscompleteness = completeness_score(y, labels) # Completeness Scoresilhouette_avg = silhouette_score(X_scaled, labels) # Silhouette Coefficientcalinski_harabasz = calinski_harabasz_score(X_scaled, labels) # Calinski-Harabasz Index# Print specific evaluation metricsprint(f'Silhouette Coefficient: {silhouette_avg:.2f}')print(f'Calinski-Harabasz Index: {calinski_harabasz:.2f}')print(f'Completeness: {completeness:.2f}')
[8/28, 12:14] ᴍᴀᴅᴅʏ: Program 10: Organizing clusters as a hierarchical treeProblem Definition:The main objective of the problem is to evaluate hierarchical clustering on a subset of the Iris dataset. Select 10 samples from each species, totaling 30 samples. Use hierarchical clustering with the Ward method to create 3 clusters. Measure the clustering performance using three metrics: completeness score (to see if samples from the same species end up in the same cluster), silhouette score (to check how well-separated the clusters are), and Calinski-Harabasz score (to assess overall clustering quality). Create a dendrogram to visualize how the clusters are formed. This will show how well the clustering matches the true species labels and provide insight into the clustering's quality.import pandas as pd # Import pandas for data manipulationfrom sklearn.preprocessing import StandardScaler # Import StandardScaler for feature scalingfrom sklearn.datasets import load_iris # Import load_iris to fetch the Iris datasetfrom sklearn.metrics import completeness_score, silhouette_score, calinski_harabasz_score # Import clustering evaluation metricsimport scipy.cluster.hierarchy as sch # Import hierarchical clustering functions from scipyimport matplotlib.pyplot as plt # Import matplotlib for plotting# Load the Iris datasetiris = load_iris() # Load the Iris dataset from sklearn# Extract features and targetX = pd.DataFrame(iris.data, columns=iris.feature_names) # Convert feature data to a pandas DataFrame with feature namesy = iris.target # Extract target labels (species) from the dataset# Combine features and target into a DataFramedata = pd.concat([X, pd.Series(y, name='species')], axis=1) # Combine features and target into a single DataFrame# Sample 10 records from each categorysample = data.groupby('species').apply(lambda x: x.sample(10, random_state=42)).reset_index(drop=True) # Sample 10 records per species from the combined DataFrame# Separate features and target in the sampleX_sample = sample.drop(columns='species') # Drop the 'species' column to get only featuresy_sample = sample['species'] # Extract the 'species' column as the target# Standardize the featuresscaler = StandardScaler() # Initialize StandardScaler for standardizing featuresX_sample_scaled = scaler.fit_transform(X_sample) # Fit and transform the features to have zero mean and unit variance# Perform hierarchical clusteringlinked = sch.linkage(X_sample_scaled, method='ward') # Perform hierarchical clustering using the Ward method, which minimizes the variance of clusters# Apply clustering to get cluster labels for a specific number of clusters (e.g., 3)num_clusters = 3 # Set the number of clusters for the final clusteringlabels = sch.fcluster(linked, num_clusters, criterion='maxclust') # Form flat clusters from the hierarchical clustering# Calculate evaluation metricscompleteness = completeness_score(y_sample, labels) # Calculate the completeness score to evaluate how well the clusters match the true labelssilhouette = silhouette_score(X_sample_scaled, labels) # Calculate the silhouette score to measure the clustering qualitycalinski_harabasz = calinski_harabasz_score(X_sample_scaled, labels) # Calculate the Calinski-Harabasz score for cluster validity# Print evaluation metricsprint(f"Number of clusters: {num_clusters}") # Print the number of clusters used in clusteringprint(f"Completeness Score: {completeness:.2f}") # Print the completeness score rounded to 2 decimal placesprint(f"Silhouette Score: {silhouette:.2f}") # Print the silhouette score rounded to 2 decimal placesprint(f"Calinski-Harabasz Score: {calinski_harabasz:.2f}") # Print the Calinski-Harabasz score rounded to 2 decimal places# Set up color mappingspecies_colors = {i: color for i, color in enumerate(plt.cm.tab10(np.linspace(0, 1, len(np.unique(y_sample)))))} # Map each species to a unique color using the tab10 colormap# Generate a dendrogram to visualize the hierarchical clusteringplt.figure(figsize=(12, 8)) # Create a figure with a specified size for the plotdendrogram = sch.dendrogram(linked,orientation='top', # Arrange the dendrogram horizontallylabels=y_sample.values, # Use the sampled target labels for the leaf labelsdistance_sort='descending', # Sort the dendrogram distances in descending ordershow_leaf_counts=True # Show the count of leaves in each cluster)# Add titles and labelsplt.title('Dendrogram of Hierarchical Clustering on Sample') # Set the title of the dendrogram plotplt.xlabel('Sample Index') # Set the x-axis labelplt.ylabel('Euclidean Distance') # Set the y-axis labelplt.show() # Display the dendrogram plot
