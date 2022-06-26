import tensorflow as tf
import numpy as np
import timeit

with tf.device('/cpu:0'):
    cpu_a = tf.random.normal([10000,1000])
    cpu_b = tf.random.normal([1000,2000])
    print(cpu_a.device, cpu_b.device)

with tf.device('/gpu:0'):
    gpu_a = tf.random.normal([10000,1000])
    gpu_b = tf.random.normal([1000,2000])
    print(gpu_a.device, gpu_b.device)
def cpu_run():
    with tf.device('/cpu:0'):
        cpu_c = tf.matmul(cpu_a,cpu_b)
    return cpu_c

def gpu_run():
    with tf.device('/gpu:0'):
        gpu_c = tf.matmul(gpu_a,gpu_b)
    return gpu_c

#warm up
cpu_time = timeit.timeit(cpu_run,number=10)
gpu_time = timeit.timeit(gpu_run,number=10)
print("warm time:" ,cpu_time ,gpu_time)

#run time
cpu_time = timeit.timeit(cpu_run,number=10)
gpu_time = timeit.timeit(gpu_run,number=10)
print("run time:" ,cpu_time ,gpu_time)

