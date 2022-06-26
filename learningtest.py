import matplotlib.pyplot as plt
import numpy as np


#L_model_forward,test
#x,parammeters = testCases.L_model_forward_test_case()
#AL,caches = mult_layer_nn.L_model_forward(x,parammeters)

#print("AL: " ,AL)
#print("")
#print("caches: " ,caches)

#compute_cost,test
#Y,aL = testCases.compute_cost_test_case()
#cost = mult_layer_nn.compute_cost(aL,Y)
#print("cost: " ,cost)

#L_model_backward,test
#AL,Y,caches = testCases.L_model_backward_test_case()

#grads = mult_layer_nn.L_model_backward(AL,Y,caches)
#print("grads: " ,grads)



x = np.linspace(-10,10,10000)
y = ((0.5*x) / (1 + np.abs(x))) + 0.5
a = 1 / (1 + np.exp(-x))


plt.plot(x,y)
plt.plot(x,a)
plt.xlabel('x')
plt.ylabel('value')
plt.show()