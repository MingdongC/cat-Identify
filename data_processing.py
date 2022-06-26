import numpy as np
import h5py


#先进行数据装载，使用lord_data_set
def lord_data_set():

    #打开训练集图片和标签
    train_dataset = h5py.File('datasets/train_catvnoncat.h5' ,"r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    #打开测试集图片和标签
    test_dataset = h5py.File('datasets/test_catvnoncat.h5' ,"r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    #两种分类
    classes = np.array(test_dataset["list_classes"][:])

    #reshape标签的维数
    train_set_y_orig = train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))

    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes


#进行数据预处理，使用pre_precessing
def pre_precessing():

    train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes =lord_data_set()

    #分别查看训练集和测试集图片的数量
    m_train = train_set_x_orig.shape[0]           #m_train是训练集图片数量
    m_test = test_set_x_orig.shape[0]             #m_test是测试集图片的数量
    print("="*10 ,"查看图片数量" ,"="*10)
    print("训练集图片的数量：" ,m_train)
    print("测试集图片的数量：" ,m_test)

    #查看每张图片的大小和维数
    train_px = train_set_x_orig.shape[1]          #train_px是训练集图片的像素
    test_px = test_set_x_orig.shape[1]            #test_px是测试集图片的像素
    print("="*7 ,"查看图片大小和维数" ,"="*8)
    print("训练集图片的大小：" ,train_px ,"*" ,train_px ,"*" ,"3")
    print("测试集图片的大小：" ,test_px ,"*" ,test_px ,"*" ,"3")
    print("训练集图片的维数：" ,train_set_x_orig.shape)
    print("训练集标签的维数：" ,train_set_y_orig.shape)
    print("测试集图片的维数：" ,test_set_x_orig.shape)
    print("测试集标签的维数：" ,test_set_y_orig.shape)

    #因为图片的维数是多维矩阵，所以需要进行降维处理
    train_set_x_orig_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
    test_set_x_orig_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
#ps:np.reshape(train_set_x_orig.shape[0],-1).T 和 np.reshape(-1,train_set_x_orig.shape[0]) 虽然矩阵维数一样但是数值是不用的

    #查看降维后的图片维数
    print("="*7 ,"查看降维后图片维数" ,"="*8)
    print("降维后训练集图片维数：" ,train_set_x_orig_flatten.shape)
    print("降维后测试集图片维数：" ,test_set_x_orig_flatten.shape)

    #将图片的特征值转换到0-1之间
    train_set_x = train_set_x_orig_flatten / 255
    test_set_x = test_set_x_orig_flatten / 255

    #处理最后的返回值变量
    train_set_y = train_set_y_orig
    test_set_y = test_set_y_orig

    return train_set_x,train_set_y,test_set_x,test_set_y,classes












