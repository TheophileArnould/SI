import numpy as np
import matplotlib as plt

class Perceptron:
    def __init__(self) -> None:
        pass

    def run(self, train_feateures, test_feateures, train_labels, test_labels,iter,alfa):
        print('Training Perceptron Neural Network')
        ##here is where all the neural network code will be

        ##lets organize the data
        Xi = np.zeros((train_feateures.shape[1] + 1 ,1)) # Vector de entradas + bias

        Wij = np.zeros((train_labels.shape[1],train_feateures.shape[1] + 1 )) # Vector de pesos

        Ai = np.zeros((train_labels.shape[1],1)) # Agreagación vector

        Yk = np.zeros((train_labels.shape[1],1)) # Vector de salidas
        
        Yd = np.zeros((train_labels.shape[1],1)) # Label Vector
        
        Ek = np.zeros((train_labels.shape[1],1)) # Error vector
        
        ecm = np.zeros((train_labels.shape[1],1)) # Ecm vector for each epoch
        
        ecmT = np.zeros((train_labels.shape[1],iter)) # Ecm result for every epoch

        ##training and testing the model

        ## Fill the weight matrix before training
        for i in range(Wij.shape[0]):
            for j in range(Wij.shape[1]):
                Wij[i,j] = np.random.rand()

        for it in range(iter):
            for iDato in range(train_feateures.shape[0]):
                Xi[0] = 1 # Bias
                for i in range(train_feateures.shape[1]):
                    Xi[i+1] = train_feateures[iDato,i]


                # let's calculate the agregation vector
                for n in range(Ai.shape[0]):
                    for n_input in range(Xi.shape[0]):
                        Ai[n] += Wij[n,n_input] * Xi[n_input]

                # let's calculate the output vector
                for n in range(Yk.shape[0]):
                    if Ai[n] < 0:
                        Yk[n] = 0
                    else:
                        Yk[n] = 1

                #let's calculate the error vector

                ##Passing training labels to Yd
                for i in range(train_labels.shape[1]):
                    Yd[i] = train_labels[iDato,i]

                for n in range(Ek.shape[0]):
                    Ek[n] = Yd[n] - Yk[n]

                    ##let's add the ecm for this data point
                    ecm[n]= ecm[n] + (Ek[n]**2)/2

                ## weight update
                for n in range(Yk.shape[0]):
                    for w in range(Wij.shape[1]):
                        Wij[n,w] = Wij[n,w] + alfa * Ek[n] * Xi[w]
                
                print(f'Iteración {it} Dato {iDato} ECM {ecm}')
                    
            for n in range(Yk.shape[0]):
                ecmT[n,iter] = ecm[n]
                ecm[n] = 0

            for n in range(Yk.shape[0]):
                plt.figure()
                plt.plot(ecmT[n,:],'r',label=f'ECM neurona {n}')
                plt.show()