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

ratingsCSV = sc.textFile("file:///home/noursaadallah/Desktop/big-data/ml-latest-small/ratings.csv")
header = ratingsCSV.first()
# remove header:
ratings = ratingsCSV.filter(lambda x : x != header)
# keep only userId and movieId columns: (u,s)
ratings = ratings.map(lambda x : x.split(',')[0]+','+x.split(',')[1])

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
# su : (s,u), p(s|u)
sz_z = sz.map(lambda x : ( x[0].split(',')[1] , x[0].split(',')[0]+','+str(x[1]))).persist()
zu_z = zu.map(lambda x : ( x[0].split(',')[0] , x[0].split(',')[1]+','+str(x[1]))).persist()

z_sz_zu = sz_z.join(zu_z)
print z_sz_zu.collect()[0:5]

# compute objective function (log-likelihood) : L = 1/N * sum over n of log(p(s|u))
