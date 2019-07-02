from sklearn.cluster import KMeans
import numpy as np
import math
import file_transfer as ft
from scipy.io import loadmat
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def mykmeans(data, target, k, iteration, ):
    '''
    Parameters
    ----------
    data: array or sparse matrix, shape (n_samples, n_features)
    target: 样本标签，shape(1,n_samples)
    k: cludter number
    iteration: 聚类最大循环次数
    return
    ----------
    result: 聚类结果和集群的失真
    '''
    model = KMeans(n_clusters=k, n_jobs=4, max_iter=iteration)  # 分为k类, 并发数4
    model.fit(data)  # 开始聚类
    # 聚类结果
    labels = model.labels_
    centers = model.cluster_centers_
    distortion = model.inertia_  #每个点到其簇的质心的距离之和。
    clusters = {}
    for i in range(k):
        clusters[i] = {}
        clusters[i]['target'] = []
        clusters[i]['data'] = []
        clusters[i]['cluster_centers'] = centers[i]
    for i in range(len(labels)):
        clusters[labels[i]]['target'].append(target[i])
        clusters[labels[i]]['data'].append(data[i])
    result = {}
    result['clusters'] = clusters
    result['distortion'] = distortion
    return result

def oldbic(n, d, distortion):
    '''
    Parameters
    ----------
    n: 样本总数量
    d: 样本维度
    distortion: 集群失真
    return
    ----------
    L: BIC 分数
    '''
    variance = distortion / (n - 1)
    p1 = -n * math.log(math.pi * 2)
    p2 = -n * d * math.log(variance)
    p3 = -(n - 1)
    L = (p1 + p2 + p3) / 2
    numParameters = d + 1
    return L - 0.5 * numParameters * math.log(n)

# 计算模型的BIC；从一个较大值的k开始，不断去除质心即减少k的数目；
# 从一个质心开始，不断分解各个簇，直到分配到各个簇的样本点符合高斯分布
def newbic(k, n, d, distortion, clustersSize):
    '''
    Parameters
    ----------
    k: 聚类数量
    n: 样本总数量
    d: 样本维度
    distortion: 集群失真
    clustersSize: 每个聚类的样本数量 shape(1,k)
    return
    logLikelihood:最大似然估计
    ----------
    L: BIC 分数
    '''
    variance = distortion / (n - k);
    L = 0.0;
    for i in range(k):
        L += logLikelihood(k, n, clustersSize[i], d, variance)
    numParameters = k + k * d;
    return L - 0.5 * numParameters * math.log(n);

def logLikelihood(k, n, ni, d, variance):
    '''
    Parameters
    ----------
    k: 聚类数量
    n: 样本总数量
    ni: 属于此聚类的样本数
    d: 样本维度
    variance: 集群的估计方差
    return
    ----------
    loglike: 后验概率估计值
    '''
    p1 = -ni * math.log(math.pi * 2);
    p2 = -ni * d * math.log(variance);
    p3 = -(ni - k);
    p4 = ni * math.log(ni);
    p5 = -ni * math.log(n);
    loglike = (p1 + p2 + p3) / 2 + p4 + p5;
    return loglike;

def myxmeans(data, target, kmin, kmax):
    '''
    Parameters
    ----------
    data: array or sparse matrix, shape (n_samples, n_features)
    target: 样本标签，shape(1,n_samples)
    k: cludter number
    iteration: 聚类最大循环次数
    return
    ----------
    result: 聚类结果和集群的失真
    '''
    d = len(data[0])
    k = kmin
    iteration = 400
    init_clusters = mykmeans(data, target, k, iteration)
    while k < kmax:
        wscc = np.zeros((k, 1))  # 每个集群的失真
        for i in range(k):
            center = init_clusters['clusters'][i]['cluster_centers']
            for tmp_sample in init_clusters['clusters'][i]['data']:
                wscc[i] += np.sqrt(np.sum(np.square(np.array(tmp_sample) - np.array(center))))
        split2cluster = {}
        for i in range(k):
            if len(init_clusters['clusters'][i]['data']) < 2:
                continue
            my2means = mykmeans(init_clusters['clusters'][i]['data'], init_clusters['clusters'][i]['target'], 2,
                                iteration)
            oldbicscore = oldbic(len(init_clusters['clusters'][i]['data']), d, wscc[i])
            newbicscore = newbic(2, len(init_clusters['clusters'][i]['data']), d, my2means['distortion'],
                                 [len(my2means['clusters'][0]['data']), len(my2means['clusters'][1]['data'])])
            if newbicscore > oldbicscore:
                split2cluster[i] = my2means
        for key in split2cluster.keys():
            init_clusters['clusters'][key] = split2cluster[key]['clusters'][0]
            init_clusters['clusters'][k] = split2cluster[key]['clusters'][1]
            k += 1
        if split2cluster == {}:
            break
    return init_clusters

def run_xmeans_semi(data):
    target = list(range(len(data)))
    # data=np.random.random((100,50))
    # target=list(range(0,100))
    xmeans_result = myxmeans(data, target, 8, 15)
    return xmeans_result

def show3D(xmeans_result):
    fig = plt.figure()
    ax = Axes3D(fig)
    all_key = []
    for key in xmeans_result['clusters'].keys():
        print('---------', key, '------------')
        print(xmeans_result['clusters'][key]['target'], '\n')
        print()
        all_key.append(key)
    centroids = []
    for j in range(len(all_key)):
        centroids.append(xmeans_result['clusters'][j]['cluster_centers'])
    c = []
    for i in range(len(all_key)):
        ax.scatter3D(np.mat(xmeans_result['clusters'][i]['data'])[:, 0],
                     np.mat(xmeans_result['clusters'][i]['data'])[:, 2],
                     np.mat(xmeans_result['clusters'][i]['data'])[:, 4])
        ax.scatter3D((centroids[i].reshape(1, 6))[:, 0], (centroids[i].reshape(1, 6))[:, 2],
                     (centroids[i].reshape(1, 6))[:, 4], marker='+', color='black')


def distance(center,point): # 计算对象到质心的距离
    #return :距离
    return np.sqrt(np.sum(np.square(np.array(point) - np.array(center))))

def getthreshold(center,data): # 计算所有对象到质心距离的平均值
    #距离的平均值
    dis_sum=0
    for tmp_sample in data:
        dis_sum += np.sqrt(np.sum(np.square(np.array(tmp_sample) - np.array(center))))
    return dis_sum/len(data)

def getthreshold1(center,data): # 计算对象到质心的最大距离
    #return:距离的最大值
    max_dis=0
    for tmp_sample in data:
        dis=np.sqrt(np.sum(np.square(np.array(tmp_sample) - np.array(center))))
        if dis>max_dis:
            max_dis=dis
    return max_dis

def transfer_file(filename):
    fr = open(filename, 'r')
    frw = open("file_new.csv", 'w')
    line = fr.readlines()
    for L in line:
        string = L.strip("\n").split(" ")

        tmp_date_list = string[0].split('-')
        temp_date = tmp_date_list[0] + tmp_date_list[1] + tmp_date_list[2]
        tmp_time_list = string[1].split(':')
        temp_time = tmp_time_list[0] + tmp_time_list[1] + tmp_time_list[2]
        a = np.float64(temp_date)
        b = np.float64(temp_time)
        c = np.float64(string[4])
        d = np.float64(string[5])
        str = '%f,%f\n' % (d, c)
        frw.write(str)
    return frw

def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

def A():
    file_name = transfer_file("data/cluster_data/dataS1.txt")
    data = loadDataSet("data/cluster_data/file_new.csv")
    xmeans_result = run_xmeans_semi(data)
    # target = list(range(len(data)))
    # xmeans_result = mykmeans(data, target, 20, 200)  # kmeans聚类
    result = {}
    key_value = {}
    all_key = []
    for key in xmeans_result['clusters'].keys():
        print('---------', key, '------------')
        print(xmeans_result['clusters'][key]['target'], '\n')
        print()
        all_key.append(key)
    every_length = []
    for m in range(len(all_key)):
        every_length.append(len(xmeans_result['clusters'][m]['target']))
    print(every_length)
    filename_front = "data/cluster_data/combination"
    filename_last = ".csv"
    nums = 1
    while nums <= 5 :
        points_normal = []
        clusters_value = []
        percent_acc = []
        each_result = {}
        acc_times = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        data_reduct = loadDataSet(filename_front+str(nums)+filename_last)
        for i in range(len(data_reduct)):
            flag = True
            for j in range(len(all_key)):
                for k in range(len(xmeans_result['clusters'][j]['data'])):
                    if data_reduct[i] == xmeans_result['clusters'][j]['data'][k]:
                        points_normal.append(xmeans_result['clusters'][j]['target'][k])
                        if j not in clusters_value:
                            clusters_value.append(j)
                        if j == 0:
                            acc_times[0] = acc_times[0] + 1
                        elif j == 1:
                            acc_times[1] = acc_times[1] + 1
                        elif j == 2:
                            acc_times[2] = acc_times[2] + 1
                        elif j == 3:
                            acc_times[3] = acc_times[3] + 1
                        elif j == 4:
                            acc_times[4] = acc_times[4] + 1
                        elif j == 5:
                            acc_times[5] = acc_times[5] + 1
                        elif j == 6:
                            acc_times[6] = acc_times[6] + 1
                        elif j == 7:
                            acc_times[7] = acc_times[7] + 1
                        elif j == 8:
                            acc_times[8] = acc_times[8] + 1
                        elif j == 3:
                            acc_times[3] = acc_times[3] + 1
                        elif j == 9:
                            acc_times[9] = acc_times[9] + 1
                        elif j == 10:
                            acc_times[10] = acc_times[10] + 1
                        elif j == 11:
                            acc_times[11] = acc_times[11] + 1
                        elif j == 12:
                            acc_times[12] = acc_times[12] + 1
                        elif j == 12:
                            acc_times[12] = acc_times[12] + 1
                        elif j == 13:
                            acc_times[13] = acc_times[13] + 1
                        elif j == 14:
                            acc_times[14] = acc_times[14] + 1
                        elif j == 15:
                            acc_times[15] = acc_times[15] + 1
                        elif j == 16:
                            acc_times[16] = acc_times[16] + 1
                        elif j == 17:
                            acc_times[17] = acc_times[17] + 1
                        elif j == 18:
                            acc_times[18] = acc_times[18] + 1
                        elif j == 19:
                            acc_times[19] = acc_times[19] + 1
                        flag = False
                        break
                    if not flag:
                        break
        for n in range(len(all_key)):
            percent_acc .append(round(acc_times[n]/every_length[n], 4))
        each_result['percent_acc'] = percent_acc
        each_result['clusters'] = sorted(clusters_value)
        # each_result['target'] = points_normal
        key_value[nums] = each_result
        nums = nums + 1
    result['No'] = key_value
    return result

if __name__ == '__main__':
    result = A()
    clusters_len = []
    print(result)
    for i in range(1,6):
        clusters_len.append(len(result['No'][i]['clusters']))
    print(clusters_len)
