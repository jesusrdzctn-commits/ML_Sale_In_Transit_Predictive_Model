# Introduction 
This code is for running the sale in transit predictive model for PMF Controllership. It runs two main processes separately: training and testing. Training shall be run every three months and testing every month but December. 

# Getting Started
Recommended Python Version 3.9.7
Install requirements.txt

# Build and Test
For training, after placing latest train data in 'Training data', type in the command window the following
        python main.py
For testing, after placing latest test data in 'input-data', type in the command window the following
        python main.py mmm-aa prediction
where mmm stands for the first three letters of the month and aa for the last two digits of the year, prediction can be 'simple' or 'doble' according to business needs.