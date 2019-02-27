# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 05:12:05 2017

@author: Administrator
"""

#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import matplotlib 

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
file_name = open("../final_project/poi_names.txt", "r")
print "Length : %d" % len (enron_data)
print len(enron_data['SKILLING JEFFREY K'])
print enron_data['SKILLING JEFFREY K'].keys()
#print enron_data['LAY KENNETH L']
#print enron_data['METTS MARK']['poi']
# Checking the NULL values in the dataset 

Poi = []
no_salary = 0
no_email = 0 
missing_totalpayments = 0
Nan_Poi_Payments = 0


for each_record in enron_data:
    Poi_yn = enron_data[each_record]['poi']
    salary = enron_data[each_record]['salary']
    email_add= enron_data[each_record]['email_address']
    total_payments = enron_data[each_record]['total_payments']
    if Poi_yn == True:
        Poi.append(Poi_yn)
    if (total_payments == 'NaN' and Poi_yn == True):
        Nan_Poi_Payments +=1
    if salary == "NaN":
        no_salary += 1 
    if email_add != "NaN":
        no_email += 1 
    if total_payments == "NaN":
        missing_totalpayments += 1 



no_poi = len(Poi)

#total_poi = 
print "Number of labelled POI"        
print no_poi 

print "Salary NOT Equal to NAN"        
print no_salary 

print "Email NOT Equal to NAN"        
print no_email

print "Total Equal to NAN"        
print missing_totalpayments




i=0
j=0
for elem in enron_data:
	#print elem
	if enron_data[elem]['total_payments'] != 'NaN' and enron_data[elem]['poi'] == True:

		i+=1
	elif enron_data[elem]['poi'] == True:
		j+=1

 
#print i 
#print j 
#j=0
#for i in file_name:
	#j+=1

#print j