# Classification-of-Biochemical-molecules-using-a-graph-kernel-based-on-Energy-and-Total-Variation
-This code classifies Biochemical molecules represented by graphs using Joint Total variation and Energy similarity measure (JET) as a kernel embeded in a Support Vector Machine. The classification process aims to determine whether the molecule is mutagenic or not against a type of DNA.  For more details and exploitation in further research work, see our published paper: 

H. Ahmed, Bay-Ahmed, Delphine Dare, and Abdel-Ouahab Boudraa. "Graph signals classification using total variation and graph energy informations.", IEEE Global Conference on Signal and Information Processing (GlobalSIP), 2017.

-In the data folder, the XML files describing molecules were preprocessed under Matlab to extract only the needed informations.
Therefore, the file "All_W_Adjacency_MUTAG.mat" contains adjacency matrices of all graphs (molecules), the file "All_NodeValues_MUTAG.mat" contains the node values of all graphs (molecules) and the file "All_Classes_MUTAG.mat" indicates the class (label) of each graph. 

-To run this program, only the mentioned three files are needed.  

-Our algorithm reached 83 % of accuracy as mentioned in the paper. Please refer to our paper in case you use this program for further research work. 

-The paper is available in:
https://sam.ensam.eu/bitstream/handle/10985/15078/IRENAV_GLOBALSIP_2017_BAYAHMED.pdf?sequence=1&isAllowed=y
