import numpy as np
import matplotlib.pyplot as plt
import data_processing

train_set_x,train_set_y,test_set_x,test_set_y,classes = data_processing.pre_precessing()

#logistic回归分析+梯度下降法模型，建立一个识别猫的网络

#数据集预处理

'''
#查看训练集和测试集图片数量
m_train_x = train_set_x_orig.shape[0]
m_test_x = test_set_x_orig.shape[0]
print("训练集图片数量：" ,m_train_x)
print("测试集图片数量：" ,m_test_x)

#查看每张图片大小和维数
num_px = train_set_x_orig.shape[1]
print("每张图片的大小" , num_px,"*" ,num_px, "* 3")
print("训练集图片的维数：" ,str(train_set_x_orig.shape))
print("训练集标签的维数：" ,str(train_set_y.shape))
print("测试集图片的维数：" ,str(test_set_x_orig.shape))
print("测试集标签的维数：" ,str(test_set_y.shape))

#将训练集和测试集图片进行降维处理
train_set_x_orig_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_orig_flatten  = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
print("降维后的训练集维度：" ,train_set_x_orig_flatten.shape)
print("降维后的测试集维度：" ,test_set_x_orig_flatten.shape)

#三原色的亮度区间是0-255
train_set_x = train_set_x_orig_flatten/255
test_set_x = test_set_x_orig_flatten/255
'''

# sigmoid函数用来求激活值
def sigmoid(z):

    a = 1/(1+np.exp(-z))
    return a

# initial_w_b函数是用来初始化w,b的值并返回w，b的值
def initial_w_b(rows_train_x):

    w = np.full((rows_train_x,1),np.random.random())
    b = 0
    assert(w.shape == (rows_train_x,1))
    assert(isinstance(b,float) or isinstance(b,int))
    return (w,b)

#定义progate函数用来正向传播和反向传播，以及求cost成本函数、dw和db
def progate(w,b,X,Y):

    m = X.shape[1]

    #先正向传播
    A = sigmoid(np.dot(w.T,X) + b)
    cost = (-1/m) * np.sum( Y*np.log(A) + (1-Y) * np.log(1-A))

    #反向传播
    dw = (1/m) * np.dot(X,(A-Y).T)
    db = (1/m) * np.sum(A-Y)

    cost = np.squeeze(cost)
    assert(dw.shape == w.shape)

    #用字典保存dw和db的值
    dictionary = {
                    "dw":dw,
                    "db":db
                }
    return (dictionary,cost)

#打印每一次迭代的权重和偏置值的变化
def print_wb_everyiter(num_iter,w,b,dw,db):
    print("="*10 ,"第" ,num_iter, "次迭代" ,"="*10)
    print("w: " ,w)
    print("b: " ,b)
    print("dw: " ,dw)
    print("db: " ,db)

#optimaize_w_b函数是用来优化w，b的值以及存储costs
def optimaize_w_b(w,b,X,Y,num_iteration,learning_rate,print_cost = False):

    costs = []

    #save_b = []
    #save_dw = []
    #save_db = []
    #save_w_mean = []

    for i in range(num_iteration):

        dictionary, cost = progate(w, b, X, Y)

        dw = dictionary["dw"]
        db = dictionary["db"]
        w = w-learning_rate*dw
        b = b-learning_rate*db
        #print_wb_everyiter(i,w,b,dw,db)


        if i%100 == 0:
            costs.append(cost)
            #save_dw.append(np.mean(dw))
            #save_db.append(db)
            #save_b.append(b)
            #save_w_mean.append(np.mean(w))
        if (print_cost == True) and (i%100 == 0):
            print("迭代次数: %d ,误差值: %f" %(i,cost))

    pramas = {
                "w":w,
                "b":b
            }

    dictionary = {
                    "dw":dw,
                    "db":db
                }
    #plt.plot(save_db)
    #plt.xlabel('iteration')
    #plt.ylabel('db')
    #plt.show()
    return (pramas,dictionary,costs)

#predict函数是使用w，b，X的值来计算预测Y_prediction并返回
def predict(w,b,X):

    #图片的数量
    m = X.shape[1]

    A = sigmoid(np.dot(w.T,X)+b)
    Y_prediction = np.zeros((1,m))

    for i in range(m):
        Y_prediction[0,i] =1 if A[0,i] >0.55 else 0

    assert(Y_prediction.shape == (1,m))
    return Y_prediction

#神经网络model
def model(x_train,y_train,x_test,y_test,num_iteration,learning_rate,print_cost = False):
    #x_train是训练集图片矩阵 --（12288×209）
    #x_test是训练集图片矩阵  --（12288×50）
    #y_train是测试集标签矩阵 --（1×209）
    #y_test是测试集标签矩阵  --（1×50）
    #num_iteration是迭代次数
    #learning_rate 是梯度下降的α

    w,b = initial_w_b(train_set_x.shape[0])
    params,dictionary,costs = optimaize_w_b(w,b,train_set_x,train_set_y,num_iteration,learning_rate,print_cost=True)

    w = params["w"]
    b = params["b"]

    Y_train_prediction = predict(w,b,x_train)
    Y_test_prediction = predict(w,b,x_test)

    assert(Y_train_prediction.shape == y_train.shape)
    assert(Y_test_prediction.shape == y_test.shape)

    print("训练集准确性：" ,(1-np.mean(np.abs(Y_train_prediction-y_train)))*100 ,"%")
    print("测试集的准确性" ,(1-np.mean(np.abs(Y_test_prediction-y_test)))*100 ,"%")

    a = {
        "costs":costs,
        "w":w,
        "b":b,
        "learning_rate":learning_rate,
        "num_iteration":num_iteration,
        "Y_train_prediction":Y_train_prediction,
        "Y_test_prediction":Y_test_prediction
    }

    return  a
a = model(train_set_x,train_set_y,test_set_x,test_set_y,num_iteration=3000,learning_rate=0.01,print_cost=True)


costs = np.squeeze(a['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(a["learning_rate"]))
plt.show()
