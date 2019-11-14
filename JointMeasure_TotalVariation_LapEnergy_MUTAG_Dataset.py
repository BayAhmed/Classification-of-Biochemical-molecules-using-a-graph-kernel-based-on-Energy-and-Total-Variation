
"""
Created on Wed Apr 26 15:14:40 2017
@author: BAY AHMED Hadj Ahmed

This code classifies molecules represented by graphs using Joint Total variation and Energy similarity measure (JET), 
embeded in a Support Vector Machine as a kernel. The classification problem is to determine whether the molecule is mutagenic or not. 
For more details and exploitation in further research work, Please refer to our paper:  
    H. Ahmed, Bay-Ahmed, Delphine Dare, and Abdel-Ouahab Boudraa. "Graph signals classification using total variation and graph energy informations." 
    2017 IEEE Global Conference on Signal and Information Processing (GlobalSIP). IEEE, 2017.
"""

import numpy as np
import scipy.io
from numpy import linalg as LA
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

# This function computes Laplacian matrix associated to a graph. It takes as entry the Adjacency matrix 
def NormLaplaceMatrix(Adj_Matrix):
    NN=np.size(Adj_Matrix,0)
    Degrees=np.sum(Adj_Matrix,0)
    
    
    Lap_Matrix=Adj_Matrix
    for i in range(NN):
        if (Degrees[0,i]!=0):
            Lap_Matrix[i,i]=1
        for j in range(NN):
            if (Lap_Matrix[i,j]!=0):
                Lap_Matrix[i,j]=-(1/(np.sqrt(Degrees[0,i]*Degrees[0,j])))
                
    return (Lap_Matrix)


# This function computes the eigenvalues of any matrix A
def OrderedEigenValues(A):     
    eigen_value, eigen_vector = LA.eigh(A)
    return(eigen_value)


# This function computes the graph Laplacian energy as defined in the paper. It takes as entries the Laplacian's
# eigenvalues, number of nodes in the graph and the number of edges.
def EigenValues_GEnergy(EigenValues_Vector,n_node,n_edge):
    
    Graph_Energy=0
    for Lamda in EigenValues_Vector:
        Graph_Energy=Graph_Energy+abs(Lamda-((2*n_edge)/n_node))
       
    return(Graph_Energy)



# This function computes the graph Laplacian energies for a set of graphs. It takes as entry an ndarray 
# containing Adjacency matrices of all graphs and an array containing nodes number associated to each graph.   
def All_LapEnergy(Adj_Graphs,Node_Values):
    N1=len(Node_Values)
    All_Graph_Energy=np.zeros((N1,1))
    for i in range(N1):
        L=Adj_Graphs[i][0]
        LL=np.asmatrix(L)
        LLL=LL.astype(float)
        Edge_n0=np.count_nonzero(LLL)
        Edge_n=Edge_n0/2
        Lap_Graph1_Norm=NormLaplaceMatrix(LLL)
        D=Node_Values[i][0]
        DD=np.asmatrix(D)
        DDD=DD.astype(float)
        Node_n=len(DDD)
        Lap_Graph1_EigenValues=OrderedEigenValues(Lap_Graph1_Norm)
        GE_1=EigenValues_GEnergy(Lap_Graph1_EigenValues,Node_n,Edge_n)
        All_Graph_Energy[i]=GE_1
    return(All_Graph_Energy)

# This function computes the total variation of a signal in a given graph. It takes as entries the Adjacency
# matrix and an array containing the signal's values associated to nodes.
def GTV(G_Signal,A):
     
     Diff_Vector=G_Signal-(np.dot(A,G_Signal))   
     G_TotalVariation=LA.norm(Diff_Vector,ord=1)
     
     return (G_TotalVariation)
 
    
# This function computes the total variation of a set of graphs. It takes as entries an ndarray containing 
# Adjacency matrices of the graphs, and an ndarray containing the graph signal of each one of them. 
def All_GTV(Adj_Graphs,Node_Values):
    N1=len(Node_Values)
    All_Graph_Totalvariation=np.zeros((N1,1))
    for i in range(N1):
        L=Adj_Graphs[i][0]
        LL=np.asmatrix(L)
        LLL=LL.astype(float)
        
        D=Node_Values[i][0]
        DD=np.asmatrix(D)
        DDD=DD.astype(float)
    
        GTV_1=GTV(DDD,LLL)
        All_Graph_Totalvariation[i]=GTV_1/len(DDD)
    return(All_Graph_Totalvariation)



# This function builts a similarity matrix between the graphs in the training set. It takes as entries: Laplacian 
# energies of all graphs, total variation of all graphs and the ponderation parameter of the joint measure. 
def SimilarityMatrix_Train(All_Graphs_GE,All_Graphs_GTV,Lambda):
    N=len(All_Graphs_GE)
    Sim_Matrix=np.zeros((N,N))
    for i in range(N):
        for j in range(N):     
               
               Diff_GE=np.absolute((All_Graphs_GE[i])-(All_Graphs_GE[j]))
               Diff_GTV=np.absolute((All_Graphs_GTV[i])-(All_Graphs_GTV[j]))
               Sim_Matrix[i][j]=np.exp(-np.absolute((Lambda*Diff_GE)+(1-Lambda)*(Diff_GTV)))
               
    return(Sim_Matrix)

# This function builts a similarity matrix between the graphs in the training set and the ones in testing set. 
# It takes as entries: Laplacian energies of all graphs (train, test), total variation of all graphs (Train, Test) and the ponderation parameter of the joint measure.
def SimilarityMatrix_Test(All_Graphs_Train_GE,All_Graphs_Test_GE,All_Graphs_Train_GTV,All_Graphs_Test_GTV,Lambda):
    N1=len(All_Graphs_Train_GE)
    N2=len(All_Graphs_Test_GE)
    Sim_Matrix=np.zeros((N2,N1))
    
    for i in range(N2):
        for j in range(N1):     
               Diff_GE=np.absolute((All_Graphs_Test_GE[i])-(All_Graphs_Train_GE[j]))
               Diff_GTV=np.absolute((All_Graphs_Test_GTV[i])-(All_Graphs_Train_GTV[j]))
               Sim_Matrix[i][j]=np.exp(-np.absolute((Lambda*Diff_GE)+(1-Lambda)*(Diff_GTV)))
               
    return(Sim_Matrix)



#%%  Data loading and preparation 

# Load from .mat files Adjacency matrices of all graphs, thier signals and their classes. You need to specify the correct paths to acces the files. 
    
AdjacencyMatrices_Dic= scipy.io.loadmat('C:\\Anaconda files\\New Kernel Molecular Graphs (Avril 2017)\\Programs\\MUTAG\\All_W_Adjacency_MUTAG.mat')
AdjacencyMatrices_Dataset=AdjacencyMatrices_Dic['All_W_Adjacency_MUTAG']

NodeValues_Dic= scipy.io.loadmat('C:\\Anaconda files\\New Kernel Molecular Graphs (Avril 2017)\\Programs\\MUTAG\\All_NodeValues_MUTAG.mat')
NodeValues_Dataset=NodeValues_Dic['All_NodeValues_MUTAG']

Classes_Dic= scipy.io.loadmat('C:\\Anaconda files\\New Kernel Molecular Graphs (Avril 2017)\\Programs\\MUTAG\\All_Classes_MUTAG.mat')
Classes_Dataset=Classes_Dic['lmutag_mod']

Graph_Energy=All_LapEnergy(AdjacencyMatrices_Dataset,NodeValues_Dataset) # Compute Laplacian energies of all graphs in dataset 
GTotal_Variation=All_GTV(AdjacencyMatrices_Dataset,NodeValues_Dataset)   # Compute Total variation of all graphs in dataset




#%% In this section, we classify the graphs using a Support Vector Machine


Accuracy_2=[]
Lambda=0.7
for j in range(10): # Repeat all operations 10 times 
    
    kf = StratifiedKFold(n_splits=10,random_state=None, shuffle=True)  # Split the data into 10 folders randomly
    Accuracy=[]
    print("Training and Testing epoch:",j+1)
    for train_index, test_index in kf.split(Graph_Energy,Classes_Dataset.ravel()): # We train on 9 folders and test in 1 folder at once.
              
           X1_train, X1_test = Graph_Energy[train_index], Graph_Energy[test_index]
           X2_train, X2_test = GTotal_Variation[train_index], GTotal_Variation[test_index]
           y_train, y_test = Classes_Dataset[train_index], Classes_Dataset[test_index]
           
           Simi_Train_Matrix=SimilarityMatrix_Train(X1_train,X2_train,Lambda)  # Build the similarity matrix between training instances 
           
           svc = svm.SVC(kernel='precomputed', C=1.0,probability=True).fit(Simi_Train_Matrix, y_train.ravel())  # Train the SVM 
                                                                     
           Simi_Test_Matrix=SimilarityMatrix_Test(X1_train,X1_test,X2_train,X2_test,Lambda) # Build the similarity matrix between training and Testing instances 
           
           Classification_Test=svc.predict(Simi_Test_Matrix) # Predict the labels of test instances 
           
           Accuracy0=metrics.accuracy_score(y_test,Classification_Test) # Compute the accuracy of prediction 
           Accuracy.append(Accuracy0)
           #print(Accuracy0)
           
    Accuracy_1=np.asarray(Accuracy)
    print("Accuracy",Accuracy_1.mean()*100,"%")
    Accuracy_2.append(Accuracy_1.mean())

Accuracy_3=np.asarray(Accuracy_2)
print("Average Accuracy=") 
print(Accuracy_3.mean()*100,"%")
print("Standard Deviation") 
print(Accuracy_3.std()*100,"%") 




