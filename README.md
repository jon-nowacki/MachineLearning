# Machine Learning
A repo of Machine Learning Programs - Demonstration purposes only.

Learning types:
 - Supervised: Prediction done at time of reqeust
 - Unsupervised: 

##
k-means clustering:
 - Description: Groups by attributes.
 - Litmus test:
 - Method: Unsupervised
 - Data type:
 - When to use:
 - Strength: Very powerful.
 - Weakness: Suffers when variance is high.
 - Test for Accuracy: Sum of squares.

k-nearest Neighbor
 - Description: 
 - Litmus test:
 - Method: 
 - Data type:
 - When to use:
 - Strength: 
 - Weakness:
 - Test for Accuracy: 

Logistical Regression
 - Description: 
 - Litmus test:
 - Method: 
 - Data type:
 - When to use:
 - Strength: 
 - Weakness:
 - Test for Accuracy: 

Multiclass Prediction
 - Description: 
 - Litmus test:
 - Method: 
 - Data type:
 - When to use:
 - Strength: 
 - Weakness:
 - Test for Accuracy: 

Regression Trees
 - Description: 
 - Litmus test:
 - Method: 
 - Data type:
 - When to use:
 - Strength: 
 - Weakness:
 - Test for Accuracy:  

Support Vector Machines
 - Description: 
 - Litmus test:
 - Method: 
 - Data type:
 - When to use:
 - Strength: 
 - Weakness:
 - Test for Accuracy: 

Niave Bayes
 - Description: P(C|A) = P(C A) / P(A) = P(C|A) dot P(A) = P(C a A)
 - Method: 
 - Litmus test: rows and column totals 
 - Data type: conditional probabilities
 - When to use:
 - Strength: 
 - Weakness: .
 - Test for Accuracy: correlated attributes


Bagging
 - Description: A modified decision tree. Ensemble method.  Mulitple independent and parallel decision trees with random samples.
 - Data type:
 - When to use:
 - Strength: Reduced variance.
 - Weakness:

Random Forests
 - Description: A modified decision tree. Ensemble method. Also called "improved bagging". Decorrelates the trees.
 - Data type:
 - When to use:
 - Strength: Reduced variance.
 - Weakness:

Boosting
 - Description: A modified decision tree. Ensemble method. Mulitple independent and sequential decision trees with original data.  Uses previous grown trees.
 - Data type:
 - When to use:
 - Strength:
 - Weakness:

Decision Trees
 - Description: Use tree structure to specify sequences of decisions and consequences.  Root, Branches, nodes and leaf nodes. Single dataset.
 - Data type: Categorical Variables.
 - When to use:
 - Strength:
 - Weakness: High variance (single sample, bias)

Regression Trees
 - Description: 
 - Litmus test:
 - Method: 
 - Data type:
 - When to use:
 - Strength: 
 - Weakness:
 - Test for Accuracy: 

These are all related:
```
decision_trees/
├── bagging
├── boosting
└── random_forests
```

Notes
1) Bagging vs Random forests:
2) Ensemble methods:
3) KNN vs Decision trees
4) Logistic Regression vs Decision Trees
5) Bagging vs Boosting

## Programs
List of programs
 - k-means_2d_cluster.py
 - k-means_multi_dimension_clustering.py


### k-means_2d_cluster.py
 
K-means clustering on a 2 dimensional grid.  X and Y coordinates only.

### k-means_multi_dimension_clustering.py

K-means clustering on a multi dimensional dataframe.
Example dataframe has 10 columns.

## Structure of Directory

```
k-means_clustering/
├── k-means_2d_cluster.py
```


#  mamba install -c anaconda seaborn
#  mamba install -c conda-forge pyodide-py
