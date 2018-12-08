# import findspark
# findspark.init()

import pyspark
import numpy as np

# example : u=10 , s=11 , z=3 => generates : [(10,11,1),(10,11,2),(10,11,3)]
def addZ (us):
    usz =[]
    for z in range(nb_z):
        usz  += [us +','+str(z)]
    return (usz)

sc = pyspark.SparkContext()
likelihood = []

ratingsCSV = sc.textFile("file:///home/noursaadallah/Desktop/big-data/ml-latest-small/ratings.csv")
header = ratingsCSV.first()
# remove header:
ratings = ratingsCSV.filter(lambda x : x != header)
# keep only userId and movieId columns: (u,s)
ratings = ratings.map(lambda x : x.split(',')[0]+','+x.split(',')[1]).persist()

_N = sc.broadcast(ratings.count() )
N = _N.value
# define number of clusters=latent classes
nb_zBC = sc.broadcast(3)
nb_z = nb_zBC.value

# generate (u,s,z)
#usz = ratings.flatMap( lambda x : addZ(x))

# p(s|z) = normalized random values
# p(z|u) = normalized random values

distinct_users = ratings.map(lambda x : x.split(',')[0] ).distinct().persist()
distinct_items = ratings.map(lambda x : x.split(',')[1] ).distinct().persist()

# generate (k,v) = ( (s,z) , p(s|z) )
sz = distinct_items.flatMap( lambda x : addZ( x ) ).map(lambda x : (x,np.random.rand())).persist()

# generate (k,v) = ( (z,u) , p(z|u) )
zu = distinct_users.flatMap( lambda x : addZ( x)).map(lambda x : (x.split(',')[1]+','+x.split(',')[0] , np.random.rand())).persist()

# normalize the probabilities generated : for each one divise by sum(p)
    # compute sums of p(z|u) and p(s|z)
sum_sz = sz.map(lambda x : x[1]).reduce(lambda x,y : x+y)
sum_zu = zu.map(lambda x : x[1]).reduce(lambda x,y : x+y)

    # divise each proba by the corresponding sum
sz = sz.map(lambda x : (x[0],x[1]/sum_sz) ).persist()
zu = zu.map(lambda x : (x[0],x[1]/sum_zu) ).persist()

# check if new sums equal 1
# sum_sz = sz.map(lambda x : x[1]).reduce(lambda x,y : x+y)
# sum_zu = zu.map(lambda x : x[1]).reduce(lambda x,y : x+y)
# print sum_sz
# print sum_zu

############################# EM #############################
# first iteration
# compute p(s|u) = Sum over z of (p(z|u) p(s|z)) 
# sz_zu : (s,z,p(s|z)), (z,u,p(z|u))
cart_prod = sz.cartesian(zu)
sz_zu = cart_prod.filter(lambda x : x[0][0].split(',')[1] == x[1][0].split(',')[0])\
    .map(lambda x : ( x[0][0].split(',')[0]+','+x[1][0].split(',')[1] , x[0][1]*x[1][1] ) )
# su : s,u p(s|u)=sum over z of p(s|z)*p(z|u)
su = sz_zu.reduceByKey(lambda x,y : x+y).persist()
# compute objective function (log-likelihood) : L = 1/N * sum over n of log(p(s|u))
_ratings = ratings.map(lambda x : (x.split(',')[1]+','+x.split(',')[0] , 0))

Lset = _ratings.join(su)

L = -Lset.map(lambda x: np.log(x[1][1])).reduce(lambda x,y : x+y)/Lset.count()
print L
likelihood.append(L)

#EM Algorithm 
#E Step: q* = (p(s/z)p(z/u))/Sum over z p(s/z)p(z/u)
