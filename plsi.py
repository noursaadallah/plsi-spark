import numpy as np
import pyspark
from numpy import random

sc = pyspark.SparkContext()
ratings = sc.textFile("file:///home/noursaadallah/Desktop/big-data/project/ratings.csv")

# split into train set and test set
# when splitting, the users and items in the test set must be in the train set
# because testing on user u50 that is not on the train set we won't have p(s|u50)
# parameters : @lamda: percentage of the test set size
#               @seed: seed of the random() function
def splitSet(ratings, lamda=0.3, seed=17):
    train_test = ratings.randomSplit([1-lamda, lamda], seed=seed)
    train, test = train_test[0], train_test[1]

    #distinct trained users and items
    trained_users = train.map(lambda x: x.split(',')[0] ).distinct().collect()
    trained_items = train.map(lambda x: x.split(',')[1] ).distinct().collect()

    #keep only users and items that are trained
    test = test.filter(lambda x: x.split(',')[0] in trained_users and x.split(',')[1] in trained_items)
    return [train,test]

# number of clusters i.e. latent classes
# according to the docs there is 19 film genres
z = sc.broadcast(5)
# generate clusters as an rdd | z as string
z_clusters = sc.parallelize(range(z.value)).map(lambda x: str(x))

# remove header
header = ratings.first()
ratings = ratings.filter(lambda x: x!= header)

#take only a sample of the dataset
ratings = ratings.sample(False, 0.1)

# split into train set and test set
train,test = splitSet(ratings)
ratings = train

# (u,s,r,t) => ((u,s);1)
ratings = ratings.map(lambda x: (x.split(',')[0]+','+x.split(',')[1] , 1) )

# broadcast the size of the dataset
N_bc= sc.broadcast(ratings.count())
N= N_bc.value

# generate distinct users and items
users = ratings.map(lambda x: x[0].split(',')[0] ).distinct()
items = ratings.map(lambda x: x[0].split(',')[1] ).distinct()

# u,s ; 0
all_us = users.cartesian(items).map(lambda x : (x[0]+','+x[1],0))
# u,s ; r       #s.t r=1 of observed u,s and 0 otherwise
usr = all_us.union(ratings).reduceByKey(lambda x,y:x+y)

#  ((u,s);r) ; z => u,s,z ; r 
uszr = usr.cartesian(z_clusters).map(lambda x: (x[0][0]+','+x[1] , x[0][1]) )

# u,s,z ; r => (u,s,z ; (r,q*))
qr = uszr.map(lambda x: (x[0] , (x[1] , random.rand()) ) )
#(u,s,z ; (r,q*)) => (u,s,z ; q*)
q = qr.map(lambda x: (x[0], x[1][1] ))
# number of partitions to be used in coalesce for each join
nb_p = q.getNumPartitions()
# array of objective function values to be plotted afterwards
LLs = []
# epsilon: error threshold
epsilon_bc = sc.broadcast(0.00001)
epsilon = epsilon_bc.value
#################################### MapReducing the model ####################################
# q*(z;u,s) = p(z|u,s) = ( N(z,s) / N(z) ) * ( p(z|u) / sum_z( N(z,s)/N(z) * p(z|u) ) )

for k in range(20): #defining a maximum number of iterations
	q.persist()

######################## M step : compute p(s|z) and p(z|u)
############ p(s|z) :
# start by computing N(z,s) = sum_u( q*(z;u,s) )
# then N(z) = sum_u_z (q*) = sum_z( N(z,s) )
# p(s|z) = N(s,z)/N(z)

# u,s,z ; q* => ( s,z ; N(s,z))     # N(s,z) = sum_u(q*)
	Nsz = q.map(lambda x: ( x[0].split(',')[1] + ',' + x[0].split(',')[2] , x[1] ) ) \
        	.reduceByKey(lambda x,y : x+y)

# ( s,z ; N(s,z)) => (z , N(z))        # N(z) = sum_s(N(s,z)) 
	Nz = Nsz.map(lambda x: ( x[0].split(',')[1] , x[1] ) ) \
        .reduceByKey(lambda x,y : x+y)

## p(s|z) = N(s,z)/N(z)
## ( (s,z), N(s,z)/N(z) )
#  ((s,z) , N(s,z)) =>  (z , (s,N(s,z)))
	Nsz = Nsz.map(lambda x: (x[0].split(',')[1] , (x[0].split(',')[0] , x[1] ) ) )

# join Nsz and Nz => (z ; ( (s,N(s,z)) , N(z) ) )
	Nsz_Nz = Nsz.join(Nz).coalesce(nb_p)
# ( (s,z) ; N(s,z)/N(z) ) 
	Psz = Nsz_Nz.map(lambda x: ( x[1][0][0]+','+x[0] , x[1][0][1] / x[1][1] ) )

############ p(z|u) :
# N(z,u) = sum_s(q*)
# N(u) = sum_s_z(q*) = sum_z( N(z,u) )
# p(z|u) = N(z,u)/N(u)
 
# u,s,z ; q* => ( u,z ; N(z,u))     # N(z,u) = sum_s(q*)
	Nzu = q.map(lambda x: (x[0].split(',')[0]+','+x[0].split(',')[2],x[1])) \
        .reduceByKey(lambda x,y: x+y)

# ( u,z , N(z,u)) => (u ; N(u))     # N(u) = sum_z( N(z,u) )
	Nu = Nzu.map(lambda x: (x[0].split(',')[0] , x[1] ) ) \
        .reduceByKey(lambda x,y: x+y)

## p(z|u) = N(z,u)/N(u)
## ( (z,u) , N(z,u)/N(u) )
# ((u,z) ; N(z,u)) => (u ; (z,N(z,u)) )
	Nzu = Nzu.map(lambda x: ( x[0].split(',')[0] , ( x[0].split(',')[1],x[1] ) ))

# join Nzu and Nu => (u ; ( (z,N(z,u)), N(u) ) )
	Nzu_Nu = Nzu.join(Nu).coalesce(nb_p)

# ( (u,z) ; p(z|u) ) # p(z|u) = N(z,u)/N(u)
	Pzu = Nzu_Nu.map(lambda x : ( x[0]+','+x[1][0][0] , x[1][0][1]/x[1][1]  )  )

######################## E step : compute q*(z;u,s) = p(z|u,s) = p(s|z) * p(z|u) / sum_z(p(s|z) * p(z|u))
## we want ( (u,s,z) ; q* )
## we have Psz = ( (s,z) ; p(s|z) )  and Pzu = ( (u,z) ; p(z|u) )

# (u,s,z ; q*) --> (u,z ; s)
	_q_ = q.map(lambda x : (x[0].split(',')[0]+','+x[0].split(',')[2] , x[0].split(',')[1]))

# (u,z ; s) join (u,z ; p(z|u)) => (u,z ; (s,p(z|u)))
	_q_ = _q_.join(Pzu).coalesce(nb_p)

# (u,z ; (s,p(z|u))) => (s,z;(u,p(z|u)))
	_q_ = _q_.map(lambda x : (x[1][0]+','+x[0].split(',')[1] , (x[0].split(',')[0],x[1][1])))

#( s,z;(u,p(z|u)) ) join (s,z; p(s|z) ) => ( s,z; (u,p(z|u),p(s|z)) )
	_q_ = _q_.join(Psz).coalesce(nb_p)

#( s,z; (u,p(z|u),p(s|z)) ) => ( u,s,z; p(z|u)*p(s|z) )
	_q_ = _q_.map(lambda x : ( x[1][0][0]+','+x[0],x[1][0][1]*x[1][1] ))

#( u,s,z; p(z|u)*p(s|z) ) => ( u,s; sum_z(p(z|u)*p(s|z)) )
	Psu = _q_.map(lambda x : (x[0].split(',')[0]+','+x[0].split(',')[1],x[1])) \
        .reduceByKey(lambda x,y : x+y)

#( u,s,z; p(z|u)*p(s|z) ) => ( u,s; (z,p(z|u)*p(s|z)) )
	_q_ = _q_.map(lambda x : (x[0].split(',')[0]+','+x[0].split(',')[1],(x[0].split(',')[2],x[1])))

# ( u,s; z,p(z|u)*p(s|z) ) join ( u,s; sum_z(p(z|u)*p(s|z)) ) => ( u,s; ( z,p(z|u)*p(s|z) ), sum_z(p(z|u)*p(s|z)) )
	_q_ = _q_.join(Psu).coalesce(nb_p)

# ( u,s; ( z,p(z|u)*p(s|z) ), sum_z(p(z|u)*p(s|z)) ) => ( u,s,z; p(z|u)*p(s|z) / sum_z(p(z|u)*p(s|z)) ) == ( u,s,z; q* )
	q = _q_.map(lambda x : ( x[0]+','+x[1][0][0], x[1][0][1]/x[1][1] ))

######################## Loglikelihood computation 
## we only maximise the probas observed in the dataset by minimising the L function (i.e (u,s) tuples such that r=1 ) 
# L = -1/N sum_n( log(p(s|u)) )

# psu join usr => u,s ; (psu , r)
	Psu_r = Psu.join(usr).coalesce(nb_p)
# filter only observed u,s : u,s ; (psu , r=1)
	Psu = Psu_r.filter(lambda x: x[1][1] == 1)
# (u,s ; psu,r ) => (u,s ; log(p(s|u)))
	log_Psu = Psu.map(lambda x: np.log(x[1][0]) )
# reduce => sum_n (log(psu))
	L = -log_Psu.reduce(lambda x,y: x+y) / N
	LLs.append(L)
	if k>1 and LLs[k-1] - LLs[k] < epsilon:
		break

## Now test the result : Psu_r are the P(s|u) => compare to test set
# u,s ; (psu , r)
psu = Psu_r

# ones are the observed tuples in the train
ones = psu.filter(lambda x: x[1][1] == 1 )
# zeroes are unobserved in the train
zeroes = psu.filter(lambda x: x[1][1] == 0 )

ones_count = ones.count()
zeroes_count = zeroes.count()

# expected : ones_mean > zeroes_mean
ones_mean = ones.map(lambda x: x[1][0]).reduce(lambda x,y: x+y) / ones_count
zeroes_mean = zeroes.map(lambda x: x[1][0]).reduce(lambda x,y: x+y) / zeroes_count

# the threshold is the mean of the observed tuples
threshold = ones_mean
# we recommand movies to users when p(s|u)>= threshold
recommended = zeroes.filter(lambda x: x[1][0] >= threshold )
recommended_count = recommended.count()

# u,s,r,t => u,s
test = test.map(lambda x: x.split(',')[0]+','+x.split(',')[1] )
# we must collect to filter the correctly recommended tuples
_test = test.collect()
recommended_correct = recommended.filter(lambda x: x[0] in _test )
print 'correctly recommended count : ' + str(recommended_correct.count())
print 'recommended count : ' + str(recommended_count)
print 'test set count : ' + str(test.count() )

############# Evaluating the recommender system
#### 1st method : precision / recall and F2
#### 2nd method : how many of the top N recommendations were correct