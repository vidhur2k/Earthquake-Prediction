# Earthquake Damage Prediction

## Introduction and Background
Within the United States alone, earthquakes destroy nearly $4.4B of economic value yearly. Our team will be delving into the 2015 Nepal Earthquake Open Data Portal. This data, which was collected using mobile devices following the Gorkha Earthquake of April 2015, details the level of destruction brought upon more than 200,000 buildings in the area. By utilizing various features reported by Nepali citizens (such as building size, purpose, and construction material), we will construct a machine learning classifier capable of determining the extent to which a building would be damaged. We will also run unsupervised clustering algorithms to search for features that best represent the damage classifications. This project, combined with technologies such as the CV-based city scanners following earthquakes (Ji, 2018) will ultimately provide a better understanding of susceptibility to earthquake-induced damage, valuable information that can be leveraged by city planners.

## Methods
This project will be divided into two parts:
1. Data Preprocessing and Visualization
2. Model Selection, Training, and Evaluation

The data has been collected for us, but we plan to spend a significant amount of time preparing (cleaning, encoding, etc) the data for the model. We will also perform meaningful visualizations to better understand the relationships between our features. The classifier will be trained using supervised learning algorithms such as Support Vector Regressors (Asim, 2018), Logistic Regression, and Hybrid Neural Networks (Asim, 2018) and unsupervised learning algorithms such as  KMeans clustering. The hyperparameters will be tuned using the GridSearchCV process. The model performances will be compared using our test set and the one that generalizes the best will be chosen.

## Expected Results
Using supervised machine learning algorithms we hope to identify which factors affect the level of damage to a building from an earthquake. We’ll compare each of the results by micro averaged F1 score, which will balance precision and recall modified to gauge accuracy for classification into 3 categories (DrivenData). We can also use dimensionality reduction to reduce the number of features from 38 to a more manageable amount by seeing which features are correlated to each other.

## Discussion
This project can benefit architects, engineers, and city planners by using the classification model to extrapolate and predict types of buildings that are likely to suffer from earthquake damage. Buildings with attributes similar to those that were more damaged can be reinforced. Both the visualization and classification models can be used in conjunction with earthquake prediction research (Rouet-Leduc, 2017) to provide advance humanitarian aid so buildings can be reinforced to take significantly less damage.

## References
Asim, K. M., Idris, A., Iqbal, T., & Martínez-Álvarez, F. (2018). Earthquake prediction model using support vector regressor and hybrid neural networks. Plos One, 13(7). doi: 10.1371/journal.pone.0199004

Rouet‐Leduc, B.,  Hulbert, C.,  Lubbers, N.,  Barros, K.,  Humphreys, C. J., &  Johnson, P. A. ( 2017).  Machine learning predicts laboratory earthquakes. Geophysical Research Letters,  44,  9276– 9282. https://doi.org/10.1002/2017GL074677 

Ji, M., Liu, L., & Buchroithner, M. (2018). Identifying Collapsed Buildings Using Post-Earthquake Satellite Imagery and Convolutional Neural Networks: A Case Study of the 2010 Haiti Earthquake. Remote Sensing, 10(11), 1689. https://doi.org/10.3390/rs10111689

DrivenData. (n.d.). Richter's Predictor: Modeling Earthquake Damage. Retrieved September 28, 2019, from https://www.drivendata.org/competitions/57/nepal-earthquake/page/136/