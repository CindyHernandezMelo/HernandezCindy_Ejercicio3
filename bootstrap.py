import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model

%matplotlib inline

data = np.loadtxt("notas_andes.dat", skiprows=1)

Y = data[:,4]
X = data[:,:4]

regresion = sklearn.linear_model.LinearRegression()

#Tamanho de subconjunto
N = 69
#Cantidad de veces que se va a correr el programa
T = 1000
# Arreglo con Betas
B = np.zeros([np.shape(X)[1], T])

for i in range(T):
    
    subconjunto_nuevo = np.random.randint(N, size=N)
    X_sub = X[subconjunto_nuevo,:]
    Y_sub = Y[subconjunto_nuevo]
    
    regresion.fit(X_sub, Y_sub)
    B[:,i] = regresion.coef_
    
    
plt.figure( )

hspace = 10 

plt.subplot(221)
plt.hist(B[0,:], bins='auto')
plt.xlim(np.min(B),np.max(B))
plt.title('$B_1$ = %.2f $\pm$ %.2f'%(np.mean(B[0,:]), np.std(B[0,:])))

plt.subplot(222)
plt.hist(B[1,:], bins='auto')
plt.xlim(np.min(B),np.max(B))
plt.title('$B_2$ = %.2f $\pm$ %.2f'%(np.mean(B[1,:]), np.std(B[1,:])))

plt.subplot(223)
plt.hist(B[2,:], bins='auto')
plt.xlim(np.min(B),np.max(B))
plt.title('$B_3$ = %.2f $\pm$ %.2f'%(np.mean(B[2,:]), np.std(B[2,:])))

plt.subplot(224)
plt.hist(B[3,:], bins='auto')
plt.xlim(np.min(B),np.max(B))
plt.title('$B_4$ = %.2f $\pm$ %.2f'%(np.mean(B[3,:]), np.std(B[3,:])))

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    

plt.savefig('')