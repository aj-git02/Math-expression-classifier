# Math-expression-classifier
Team - BugML Team members - Akarsh Jain,Kushagra Rode,Atharv Dabli <br/>
Details of the libraries required are in requirements.txt  <br/>
This is a ML model to make math easy (just a overkill) :)  <br/>
# About the repository  
Task 1 of the project has been completed with 95% accuracy (approx) <br/>
implementation details given as comments  <br/>
Task 2 : unable to complete successfully - details of the attempt are as follows and also given at last as comments tried to implement three CNN's each for prefix, postfix and infix (these are classified by earlier CNN) did not work and the network seemed to be taking out the average of the labels(values) and outputing this for each image tried varying different hyper-parameters for 2-3 days wide range of values but to no use this error seemed to be due to the less informative labels Hence then tried a K means algorithm to classify *,/,+,- which also did not work (tried to divide in 4 clusters)  <br/>
To classify your math equations run inference1.py in the terminal with giving input as the path to the image/s. This will generate BugML_1.csv file in the same directory with your answers  <br/>
Future Prospects - DeBugML <br/>
