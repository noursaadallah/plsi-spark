import matplotlib.pyplot as plt

# LLs is the list of loglikelihood values
LLs = [1.63213750157692, 1.6105495618583288, 1.6105408571008568, 1.6105322787614287, 1.6105238241956794, 1.6105154908344097, 1.6105072761808805, 1.6104991778082138, 1.6104911933569128, 1.6104833205324827, 1.6104755571031546, 1.6104679008977034, 1.6104603498033583, 1.6104529017637994, 1.610445554777235, 1.6104383068945602, 1.6104311562175884, 1.610424100897355, 1.6104171391324857, 1.610410269167638, 1.610403489291994, 1.6103967978378193, 1.6103901931790765, 1.6103836737300903, 1.6103772379442667, 1.6103708843128608, 1.610364611363789, 1.6103584176604901, 1.6103523018008254, 1.61034626241602]
plt.plot(LLs)
plt.ylabel('loglikelihood')
plt.xlabel('iteration')
plt.title('Log Likelihood = f(iterations)')
plt.show()