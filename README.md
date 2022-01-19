# Credit_Risk_Analysis
Analyzing the risk of loans with several machine learning models

## Overview of the analysis
In this analysis we used Python to create and implement supervised Machine Learning techniques that analyzed a dataset in order to predict credit risk for potential clients. This analysis was performed on a dataset comprised of individuals who had acquired loans and was made up of demographic information, loan status, financial history, etc.  

Since there is inherently a much lower number of defaulted loans in our data set compared to non-defaulted loans, the groups were hugely imbalanced. In order to address this discrepancy in our analysis, we used various models to balance the groups and then compared their accuracy in predicting risk. We employed and compared the following supervised Machine Learning Models:  
 - Oversampling with the <b> RandomOverSampling</b> and <b> SMOTE</b> algorithms. 
 - Undersampling with the <b> ClusterCentroids</b> algorithms. 
 - Combination approach with both oversampling and under-sampling using the <b> SMOTEENN</b> algorithm. 
 - We then used two additional models that reduce bias in prediction. The first was <b> BalancedRandomForestClassifier</b> and the second was <b> EasyEnsembleClassifier</b>. 

## Results (Balanced Accuracy Scores, Confusion Matrices, and Imbalanced Classification Reports) 

### Random Oversampling Model

![Naive_Oversampling_cm_bas.png](Resources/Naive_Oversampling_cm_bas.png)
![Naive_Oversampling_cs.png](Resources/Naive_Oversampling_cs.png)
The balanced accuracy score was 65%. 
The high_risk precision is about 1% only with 74% sensitivity which makes a F1 of 2% only.
Due to the high number of the low_risk population, its precision is almost 100% with a sensitivity of 56%.

### SMOTE Oversampling Model

![SMOTE_cm_bas.png](Resources/SMOTE_cm_bas.png)
![SMOTE_cr.png](Resources/SMOTE_cr.png)
These results are similar to the first oversampling model above. 
The balanced accuracy score was 66%. 
The high_risk precision is about 1% only with 62% sensitivity which makes a F1 of 2%.
Due to the high number of the low_risk population, its precision is almost 100% with a sensitivity of 69%.

### ClusterCentroid Undersampling Model

![ClusterCentroid_Undersamplig_cm-bas.png](Resources/ClusterCentroid_Undersamplig_cm-bas.png)
![ClusterCentroid_Undersamplig_cr.png](Resources/ClusterCentroid_Undersamplig_cr.png)
The balanced accuracy score reduced to 54%. 
The high_risk precision is still 1% with 69% sensitivity. This makes a F1 of only 1%.
Due to the high number of false positive results, the sensitivity for low_risk is 40%, though the precision is high.

### SMOTEENN Combination Sampling Model

![SMOTEEN_cm_bas.png](Resources/SMOTEEN_cm_bas.png)
![SMOTEEN_cr.png](Resources/SMOTEEN_cr.png)
The balanced accuracy score was 65%, similar to the oversampling models. 
The high_risk precision is 1% with 72% sensitivity which makes a F1 of 2%, similar to the oversampling models.
The number of the false positive in the low_risk population is lower so the sensitivity increased to 57% from the undersampling model.

### Blanced Random Forest Classifier Model

![BalancedRFC_bas.png](Resources/BalancedRFC_bas.png)
![BalancedRFC_cm.png](Resources/BalancedRFC_cm.png)
![BalancedRFC_cr.png](Resources/BalancedRFC_cr.png)
The balanced accuracy score was now higher at 79%. 
The high_risk precision remains low at 3% with a 70% sensitivity. This makes a slightly higher F1 of 6%.
Due to a smaller number of false positives, the low_risk precision is 100% with a sensitivity of 87%. This is an improvement over the previous models. 

### Easy Ensemble Classifier Model 

![EasyEClassifier_bas.png](Resources/EasyEClassifier_bas.png)
![EasyEClassifier_cm.png](Resources/EasyEClassifier_cm.png)
![EasyEClassifier_cr.png](Resources/EasyEClassifier_cr.png)
The balanced accuracy score is very high at 93%. 
The high_risk precision remains low at 9% with a 92% sensitivity. This results in the highest F1 of 16%.
There are fewer false positives, the low_risk precision is 100% with a sensitivity of 94%. This is a huge improvement over the previous models. 

## Summary
In our analysis, we found that all the models had low precision when it came to the high credit risk group. The highest precision rating for this group was 9% with the Ensemble method. This is still a very low precision rating, though the recall or sensitivity was high at 92%. This means that while nearly all of the high-risk credit individuals were identified, there are a lot of low risk credit individuals who were incorrectly categorized. This would mean that these incorrectly categorized individuals would not be given loans and the back would miss out on the opportunity for additional clients, and therefore additional revenue. Based on our analysis, I would say that additional modeling algorithms should be performed and evaluated to try to identify more accurate models. In the absence of additional modeling, I would use the Easy Ensemble Classifier model, and keep in mind the noted limitations it contains.
