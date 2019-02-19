#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 18:11:45 2018

@author: Huilin
"""


from sklearn.ensemble import RandomForestClassifier
class mnistClassifier(RandomForestClassifier):
        
    def score(self,X,y):       
       
        return super().score(X, y)