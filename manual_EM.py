import numpy as np 
import pyspark
from numpy import random

sc = pyspark.SparkContext()
# number of clusters
nb_z = sc.broadcast(3)

# A parser for extracting the userId and itemId columns
def extract_us(line):
    line = line.split(',')
    line = line[0]+','+line[1]
    return(line)

# add values of z to a (u,s) tuple
def suffix_z(us):
    res = []
    for z in range(nb_z.value):
        res += [us+','+str(z)]
    return(res)

# split into train set and test set
# when splitting the users and items in the test set must be in the train set
# because testing on user ex50 that is not on the train set we won't have p(s|ex50)
def splitSet(ratings, lamda=0.2, seed=17):
    train_test = ratings.randomSplit([1-lamda, lamda], seed=seed)
    train, test = train_test[0], train_test[1]
    #broadcast the trained users and items to keep a common reference among the nodes
    trained_users = sc.broadcast(train.map(lambda x: x.split('')[0] ).distinct())
    trained_items = sc.broadcast(train.map(lambda x: x.split('')[1] ).distinct())
    #keep only users and items that are trained
    test = test.filter(lambda x: x.split(',')[0] in trained_users.value and x.split(',')[1] in trained_items.value)
    return [train,test]


ratings = sc.textFile("/home/noursaadallah/Desktop/big-data/project/ratings.csv")
# remove header
header = ratings.first()
ratings = ratings.filter(lambda x: x != header )

# create (k,v) tuples ((u,s,z), q*). q* is random
# q* is not necessarily a normalized probability at first, 
# it will be when EM starts because it's sum/sum => the measure of the proba will become 1
q0 = ratings.map(extract_us).flatMap(suffix_z).map(lambda usz : (usz, random.rand())).persist()

#################################### MapReducing the model ####################################
# q*(z;u,s) = p(z|u,s) = ( N(z,s) / N(z) ) * ( p(z|u) / sum_z( N(z,s)/N(z) * p(z|u) ) )
######################## M step : compute p(s|z) and p(z|u)

############ p(s|z) :
# start by computing N(z,s) = sum_u( q*(z;u,s) )
# then N(z) = sum_u_z (q*) = sum_z( N(z,s) )
# p(s|z) = N(s,z)/N(z)

# ((s,z) , N(s,z))
# N(s,z) is the sum over users of q*
Nsz = q0.map(lambda x: ( x[0].split(',')[1] + ',' + x[0].split(',')[2] , x[1] ) ) \
        .reduceByKey(lambda x,y : x+y).persist()

# (z , N(z))
# N(z) is the sum over items of N(s,z)
Nz = Nsz.map(lambda x: ( x[0].split(',')[1] , x[1] ) ) \
        .reduceByKey(lambda x,y : x+y)

## p(s|z) = N(s,z)/N(z)
## ( (s,z), N(s,z)/N(z) )
# map ((s,z) , N(s,z)) to (z , (s,N(s,z)))
Nsz = Nsz.map(lambda x: (x[0].split(',')[1] , (x[0].split(',')[0] , x[1] ) ) )

# join Nsz and Nz => (z , ( (s,N(s,z)) , N(z) ) )
Nsz_Nz = Nsz.join(Nz)
# ( (s,z) , N(s,z)/N(z) ) 
Psz = Nsz_Nz.map(lambda x: ( x[1][0][0]+','+x[0] , x[1][0][1] / x[1][1] ) )

############ p(z|u) :
# N(z,u) = sum_s(q*)
# N(u) = sum_s_z(q*) = sum_z( N(z,u) )
# p(z|u) = N(z,u)/N(u)
 
# ((u,z) , N(z,u))
# N(z,u) is the sum over the items of q*
Nzu = q0.map(lambda x: (x[0].split(',')[0]+','+x[0].split(',')[2],x[1])) \
        .reduceByKey(lambda x,y: x+y).persist()

# u , N(u)
# N(u) = sum over z of N(z,u)
Nu = Nzu.map(lambda x: (x[0].split(',')[0] , x[1] ) ) \
        .reduceByKey(lambda x,y: x+y)

## p(z|u) = N(z,u)/N(u)
## ( (z,u) , N(z,u)/N(u) )
# map ((u,z),N(z,u)) to (u, (z,N(z,u)) )
Nzu = Nzu.map(lambda x: ( x[0].split(',')[0] , ( x[0].split(',')[1],x[1] ) ))

# join Nzu and Nu => (u, ( (z,N(z,u)), N(u) ) )
Nzu_Nu = Nzu.join(Nu)

# ( (u,z) , p(z|u) ) # p(z|u) = N(z,u)/N(u)
Pzu = Nzu_Nu.map(lambda x : ( x[0]+','+x[1][0][0] , x[1][0][1]/x[1][1]  )  )

######################## E step : compute q*(z;u,s) = p(z|u,s) = p(s|z) * p(z|u) / sum_z(p(s|z) * p(z|u))
## we want ( (u,s,z) ; q* )
## we have Psz = ( (s,z) ; p(s|z) )  and Pzu = ( (u,z) ; p(z|u) )

# (u,s,z ; q*) --> (u,z ; s)
_q0_ = q0.map(lambda x : (x[0].split(',')[0]+','+x[0].split(',')[2] , x[0].split(',')[1])).persist()

# (u,z ; s) join (u,z ; p(z|u)) => (u,z ; (s,p(z|u)))
_q0_ = _q0_.join(Pzu)

# (u,z ; (s,p(z|u))) => (s,z;(u,p(z|u)))
_q0_ = _q0_.map(lambda x : (x[1][0]+','+x[0].split(',')[1] , (x[0].split(',')[0],x[1][1])))

#( s,z;(u,p(z|u)) ) join (s,z; p(s|z) ) => ( s,z; (u,p(z|u),p(s|z)) )
_q0_ = _q0_.join(Psz)

#( s,z; (u,p(z|u),p(s|z)) ) => ( u,s,z; p(z|u)*p(s|z) )
_q0_ = _q0_.map(lambda x : ( x[1][0][0]+','+x[0],x[1][0][1]*x[1][1] ))

#( u,s,z; p(z|u)*p(s|z) ) => ( u,s; sum_z(p(z|u)*p(s|z)) )
sum_q0_ = _q0_.map(lambda x : (x[0].split(',')[0]+','+x[0].split(',')[1],x[1])).reduceByKey(lambda x,y : x+y)

#( u,s,z; p(z|u)*p(s|z) ) => ( u,s; z,p(z|u)*p(s|z) )
_q0_ = _q0_.map(lambda x : (x[0].split(',')[0]+','+x[0].split(',')[1],(x[0].split(',')[2],x[1])))

# ( u,s; z,p(z|u)*p(s|z) ) join ( u,s; sum_z(p(z|u)*p(s|z)) ) => ( u,s; ( z,p(z|u)*p(s|z) ), sum_z(p(z|u)*p(s|z)) )
_q0_ = _q0_.join(sum_q0_)

# ( u,s; ( z,p(z|u)*p(s|z) ), sum_z(p(z|u)*p(s|z)) ) => ( u,s,z; p(z|u)*p(s|z) / sum_z(p(z|u)*p(s|z)) ) == ( u,s,z; q* )
q1 = _q0_.map(lambda x : ( x[0]+','+x[1][0][0], x[1][0][1]/x[1][1] ))