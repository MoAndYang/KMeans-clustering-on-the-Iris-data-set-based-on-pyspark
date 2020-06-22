from pyspark import SparkContext
from pyspark.sql import SparkSession,Row
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":
    sc = SparkContext(appName="KMeans_pyspark",master='local')  # SparkContext
    
    #----读取并处理数据----
    print("------------------读取数据-----------------")
    
    SparkSession(sc)   #利用SparkSession来使sc具有处理PipelinedRDD的能力
    
    def f(x):
        rel = {}
        rel['iris_features'] = Vectors.dense(float(x[0]),float(x[1]),float(x[2]),float(x[3]))
        return rel
    iris_DF = sc.textFile('./iris.txt').map(lambda line:line.split('\t')).map(lambda p:Row(**f(p))).toDF()  # transform RDD to DataFrame
    iris_DF.show()
        
    #----Kmeans聚类----
    print("------------------Kmeans聚类--------------------")
    print("------------设定不同的K值，进行分类,计算平方误差之和------------")
    
    errors = []
    results = []
    centers = []
    
    for k in range(2,10):
        kmeansmodel = KMeans().setK(k).setFeaturesCol('iris_features').setPredictionCol('prediction').fit(iris_DF)
   
        print("With K={}".format(k))
        
        #带有预测簇标签的数据集
        kmeans_results = kmeansmodel.transform(iris_DF).collect()
        results.append(kmeans_results)
        for item in kmeans_results:
            print(str(item[0])+' is predcted as cluster'+ str(item[1]))
        
        #获取到模型的所有聚类中心情况
        kmeans_centers = kmeansmodel.clusterCenters()
        centers.append(kmeans_centers)
        center_seq = 0
        for item in kmeans_centers:
            print("Cluster" +  str(center_seq) + "  Center" + str(item))
            center_seq = center_seq + 1
      
        #计算集合内误差平方和（Within Set Sum of Squared Error, WSSSE)
        WSSSE = kmeansmodel.computeCost(iris_DF)
        errors.append(WSSSE)
        print("Within Set Sum of Squared Error = " + str(WSSSE))
        
        print('--'*30 + '\n')
        
    #----WSSSE可视化----
    plt.figure()
    k_number = range(2,10)
    plt.plot(k_number,errors)
    plt.xlabel('Number of K')
    plt.ylabel('WSSSE')
    plt.title('K-WSSSE')
  
    
    #----聚类结果可视化----
    print("---------将数据转换为panda结构，并查看空间3d图心-----------")
     #通过K-WSSSE图，k=6时聚类效果较好
    k = 6

    cluster_vis = plt.figure(figsize=(10,10)).gca(projection='3d')
    
    for item in results[k-2]:
        if item[1] == 0:
            cluster_vis.scatter(item[0][0],item[0][1],item[0][2],c = 'b') # blue
        if item[1] == 1:
            cluster_vis.scatter(item[0][0],item[0][1],item[0][2],c = 'y') # yellow
        if item[1] == 2:
            cluster_vis.scatter(item[0][0],item[0][1],item[0][2],c = 'm') # magenta
        if item[1] == 3:
            cluster_vis.scatter(item[0][0],item[0][1],item[0][2],c = 'k') # black
        if item[1] == 4:
            cluster_vis.scatter(item[0][0],item[0][1],item[0][2],c = 'g') # green
        if item[1] == 5:
            cluster_vis.scatter(item[0][0],item[0][1],item[0][2],c = 'c') # cyan

   
    for item in centers[k-2]:
            cluster_vis.scatter(item[0],item[1],item[2],c = 'r',marker = 'p') # red,五角
             
    plt.show()
