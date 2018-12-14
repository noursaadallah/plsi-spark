import numpy as np 
import pyspark
from numpy import random

########################################## Check if the splitSet function behaves properly ##########################################

# split into train set and test set
# when splitting, the users and items in the test set must be in the train set
# because testing on user u50 that is not on the train set we won't have p(s|u50)
# parameters : @lamda: percentage of the test set size
#               @seed: seed of the random() function
def splitSet(ratings, lamda=0.3, seed=17):
    train_test = ratings.randomSplit([1-lamda, lamda], seed=seed)
    train, test = train_test[0], train_test[1]
    #broadcast the trained users and items to keep a common reference among the nodes
    trained_users = train.map(lambda x: x.split(',')[0] ).distinct()
    trained_items = train.map(lambda x: x.split(',')[1] ).distinct()
    
    trained_users = sc.broadcast(trained_users.collect()).value
    trained_items = sc.broadcast(trained_items.collect()).value

    #keep only users and items that are trained
    test = test.filter(lambda x: x.split(',')[0] in trained_users and x.split(',')[1] in trained_items)
    return [train,test]

sc = pyspark.SparkContext()

ratings = sc.textFile("/home/noursaadallah/Desktop/big-data/project/ratings.csv")
# remove header
header = ratings.first()
ratings = ratings.filter(lambda x: x != header )

train,test = splitSet(ratings)

print 'full set size : ' +  str(ratings.count())
print 'train set size : ' + str(train.count())
print 'test set size : ' +  str(test.count())