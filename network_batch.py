import numpy as np
import matplotlib.pyplot as plt
import fb_functions_dnn
import json
import sys


##定义Quadratic_Cost为二次代价函数类,包括成本输出方法和计算后向传播的dA的方法
class Quadratic_Cost:

    @staticmethod
    def function(A,Y):
        '''
            :param A: A是前向传播的最后激活值
            :param Y: Y是数据集标签
            :param m: m是样本数量
            :param cost: cost是一次迭代的成本值
            :return: 返回一次迭代的成本值
        '''
        m = Y.shape[1]
        cost = (1/(2*m)) * (np.linalg.norm(A-Y)**2)
        return cost
    @staticmethod
    def derivative(A,Y):
        ##dA是最后一层后向传播的梯度值
        dA = A-Y
        return dA

##定义Cross_Entropy_Cost为交叉熵代价函数，包括成本输出方法和计算后向传播dA的方法
class Cross_Entropy_Cost:

    @staticmethod
    def function(A,Y):
        '''
            :param A: A是前向传播的预测值
            :param Y: Y是数据集的标签
            :param m: m是样本数量
            :return: 返回一次迭代的成本值
        '''
        m = Y.shape[1]
        cost = (1/m) * np.sum(-np.multiply(np.log(A),Y) - np.multiply(np.log(1-A),1-Y))
        return cost
    @staticmethod
    def derivative(A,Y):
        ##dA是最后一层后向传播的梯度值
        dA = -(np.divide(Y, A)) + (np.divide((1 - Y), (1 - A)))
        return dA

class Network_batch:

    #自动调用初始化函数
    def __init__(self,layers_dims,cost=Cross_Entropy_Cost):
        '''
            layers_dims: 是一个记录神经网络的hiddenlayers和每层的units的数组
            parameters: 是一个保存各层“w”和“b"值的字典
            caches: 是用来保存每层的cache_linear和cache_activation
            cache_linear: 保存前向传播的A_pre,w,b值
            cache_activation: 保存前向传播的A激活值
            grads_backward: 保存后向传播的梯度值，dA,dw,db
            costs: 迭代的成本值
        '''
        self.grads_backward = {}
        self.caches = []
        self.costs = []
        self.num_layers = len(layers_dims)
        self.layers_dims = layers_dims
        self.default_wb_initializer()
        self.cost = cost

    #定义默认初始化权重和偏置的函数
    def default_wb_initializer(self):

        #biases和weigts分别是包含初始化后各层的偏置和权重值，采用高斯分布初始化

        np.random.seed(1)
        self.weights = [0] * self.num_layers
        self.biases = [0] * self.num_layers

        for l in range(1,self.num_layers):
            self.weights[l] = np.random.randn(self.layers_dims[l],self.layers_dims[l-1]) * np.sqrt(2/self.layers_dims[l-1])
            self.biases[l] = np.zeros((self.layers_dims[l],1))

            assert(self.weights[l].shape == (self.layers_dims[l],self.layers_dims[l-1]))
            assert(self.biases[l].shape == (self.layers_dims[l],1))

    #前向线性传播linear_forward
    def linear_forward(self,A_pre,l):

        '''
            z为前向传播的计算值
            cache保存上层传过来的输入值，和本层的w，b值
        '''

        z = np.dot(self.weights[l],A_pre) + self.biases[l]
        cache_linear = (A_pre,self.weights[l],self.biases[l])

        assert(z.shape == (self.weights[l].shape[0],A_pre.shape[1]))

        return z,cache_linear

    #前向传a播激活函数linear_activation_forward
    def linear_activation_forward(self,z,cache_linear,activation_forward = "relu_forward"):

        '''
            param z: 线性传播的计算值
            param A: 本层的激活值
            param activation_function: 本层使用的激活函数
            cache: 保存A_pre,w,b以及本层的前向传播计算值
            return: 返回本层的激活值以及cache
        '''

        if activation_forward == "relu_forward":
            A,cache_activation = fb_functions_dnn.relu_forward(z)
        elif activation_forward == "sigmoid_forward":
            A,cache_activation = fb_functions_dnn.sigmoid_forward(z)
        elif activation_forward == "g_softsign_forward":
            A,cache_activation = fb_functions_dnn.g_softsign_forward(z)

        assert(A.shape == z.shape)
        cache = (cache_linear,cache_activation)

        return A,cache

    #L层的前向传播model，L_model_forward
    def model_forward(self,x):

        A = x

        for l in range(1,self.num_layers-1):
            z,cache_linear = self.linear_forward(A,l)
            A,cache_activation = self.linear_activation_forward(z,cache_linear,"relu_forward")
            cache = (cache_linear,cache_activation)
            self.caches.append(cache)
        z,cache_linear = self.linear_forward(A,self.num_layers-1)
        AL,cache_activation = self.linear_activation_forward(z,cache_linear,"sigmoid_forward")
        cache = (cache_linear,cache_activation)
        self.caches.append(cache)
    
        return AL

    # 计算成本函数compute_cost
    def compute_cost(self,AL,Y):

        '''
            计算一次迭代的总成本值
        '''
        return self.cost.function(AL,Y)

    #后向线性传播函数linear_backward
    def linear_backward(self,dz,A_pre,w,b):

        '''
            param m: 训练集样本数量
            param dw: 本层w的梯度值
            param db: 本层b的梯度值
            param dA_pre: 后向传播下一层的梯度值
        '''

        m = A_pre.shape[1]
        dw = (1/m) * np.dot(dz,A_pre.T)
        db = (1/m) * np.sum(dz)
        dA_pre = np.dot(w.T,dz)

        assert(dw.shape == w.shape)
        assert(dA_pre.shape == (A_pre.shape[0],m))

        return dA_pre,dw,db

    #后向线性传播激活函数linear_activation_backward
    def linear_activation_backward(self,dA,l,activation_backward = "relu_backward"):

        '''
            param dA: 本层后向传播的梯度值
            param l: 第l层前向传播模块cache的值
            param dA_pre: 后向传播模块下一层的梯度值
            param dw: w的梯度值
            param db: b的梯度值
        '''

        cache_linear,cache_activation = self.caches[l]
        A_pre,w,b = cache_linear
        A,z = cache_activation
        dA_pre = np.zeros(A_pre.shape)
        dw = np.zeros(w.shape)
        db = np.zeros(b.shape)

        if activation_backward == "relu_backward":
            dz = fb_functions_dnn.relu_backward(dA,z)
            dA_pre,dw,db = self.linear_backward(dz,A_pre,w,b)
        elif activation_backward == "sigmoid_backward":
            dz = fb_functions_dnn.sigmoid_backward(dA,z)
            dA_pre,dw,db = self.linear_backward(dz,A_pre,w,b)
        elif activation_backward == "g_softsign_backward":
            dz = fb_functions_dnn.g_softsign_backward(dA,z)
            dA_pre,dw,db = self.linear_backward(dz,A_pre,w,b)

        return dA_pre,dw,db

    #L层后向传播model，L_model_backward
    def model_backward(self,AL,Y):

        '''
        param grads_backward[dA + l]: l为当前的层数，dAl保存的是当前层向后传播的dA值，即dA2为第二层的dA_pre
        param AL: 前向传播最后一层的激活值
        param Y: 数据集的标签
        param caches: 保存了前向传播每一层的cache_linear(A_pre,w,b)和cache_activation(A,z)
        return: 返回一个字典grads_backward保存每一层的dA,dw，db
        '''

        AL = AL.reshape(Y.shape)                                          #求dAL之前保证AL和Y的shape是一样的
        dAL = self.cost.derivative(AL,Y)                                  #dAL是成本函数对al的偏导
        self.grads_backward["dA" + str(self.num_layers-1)],self.grads_backward["dw" + str(self.num_layers-1)],self.grads_backward["db" + str(self.num_layers-1)] = \
            self.linear_activation_backward(dAL,self.num_layers-2,"sigmoid_backward")
        for l in reversed(range(self.num_layers-2)):
            dA_temp,dw_temp,db_temp = self.linear_activation_backward(self.grads_backward["dA" + str(l+2)],l,"relu_backward")
            self.grads_backward["dA" + str(l+1)] = dA_temp
            self.grads_backward["dw" + str(l+1)] = dw_temp                       #caches保存的L个参数是从0到L-1
            self.grads_backward["db" + str(l+1)] = db_temp                       #循环是从L-2开始往后到0，故有l+1，和l+2

    #利用后向传播的求导值更新参数updata_parameters
    def updata_parameters(self,learning_rate):

        '''
            param parameters:   包含w,b参数的字典
            param grads_backward:  包含梯度值的字典
            param learning_rate:    学习效率
            return: 更新后的parameters
        '''

        lenth = self.num_layers
        for l in range(lenth-1):
            self.weights[l+1] = self.weights[l+1] - learning_rate * self.grads_backward["dw" + str(l+1)]
            self.biases[l+1] = self.biases[l+1] - learning_rate * self.grads_backward["db" + str(l+1)]

    #打印每一次迭代的权重和偏置值的变化
    def print_wb_everyiter(self,num_iter):
        print("="*20 ,"第" ,num_iter, "次迭代" ,"="*20)
        for i in range(1,self.num_layers+1):
            print("w" ,i ,": " ,self.weights[i] ,"b" ,i ,": " ,self.biases[i])

    #L层神经网络模型L_layers_model
    def Batch_GD(self,X,Y,num_iteration=3000,learning_rate=0.005,print_cost=False,isplot=False):
        '''
        :param X: 数据集图像
        :param Y: 数据集标签
        :param num_iteration: 迭代次数
        :param learning_rate: 学习率
        :param print_cost: 是否打印
        :param isplot: 是否显示图像
        '''

        for i in range(num_iteration):
            AL = self.model_forward(X)
            cost = self.compute_cost(AL,Y)
            self.model_backward(AL,Y)
            self.updata_parameters(learning_rate)
            if i % 100 == 0:
                self.costs.append(cost)
                if print_cost:
                    print("迭代次数：" ,i ,"成本值：" ,cost)
        if isplot:
            plt.plot(np.squeeze(self.costs))                                  #plot是需要打印的值
            plt.ylabel('cost')                                          #ylabel是纵坐标
            plt.xlabel('iteration (per hundreds)')                          #xlabel是横坐标
            plt.title("learning rate = " + str(learning_rate))          #title是标题
            plt.show()

    #利用训练好的网络做预测predict
    def predict(self,X,Y):

        m = Y.shape[1]
        Y_prediction = np.zeros((Y.shape))
        AL = self.model_forward(X)
        for i in range(0,m):
            if AL[0,i] > 0.5:                       #0.5为图片输出的阈值，大于这个值就输出为1（cat），小于就输出为0（non-cat）
                Y_prediction[0,i] = 1
            else:
                Y_prediction[0,i] = 0
        Y_prediction = Y_prediction.reshape(Y.shape)

        return Y_prediction

    #定义计算准确性的函数accuracy
    def accuracy(self,X,Y,data_set = "train_set"):

        Y_prediction = self.predict(X,Y)
        if data_set == "train_set":
            print("训练集准确性：", (1 - np.mean(np.abs(Y_prediction - Y))) * 100, "%")
        elif data_set == "test_set":
            print("测试集准确性：", (1 - np.mean(np.abs(Y_prediction - Y))) * 100, "%")

    #定义储存网络的函数save
    def save(self,filename):
        data = {
            "layers_dims" : self.layers_dims,
            "weights" : self.weights,
            "biases" : self.biases,
            "cost" : str(self.cost)
        }
        f = open(filename,"w")
        json.dump(data,f)
        f.close()

##定义load函数load a network
def load(filename):
    '''
    :param filename:
    :return:
    '''
    f = open(filename,"r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__],data["cost"])
    net = Network_batch(data["layers_dims"],cost)
    net.weights = data["weights"]
    net.biases = data["biases"]

    return net
