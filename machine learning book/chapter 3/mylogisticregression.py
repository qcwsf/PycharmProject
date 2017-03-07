
import numpy as np

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

class logRegressClassifier(object):
    def __init__(self):
        self.dataMat = np.mat(()) #11
        self.labelMat = np.mat(())
        self.weights = np.mat(())

    def train(self):
        m, n = np.shape(self.dataMat)
        for i in range(20):
            self.NewtonAscent()
            print self.logmle()
            # print self.weights
        classify_tag=np.zeros((m,1))
        for i in range(m):
            classify_tag[i]=self.classify(self.dataMat[i].transpose())
        print classify_tag


    def classify(self, x):
        prob = sigmoid(self.weights.transpose()*x)
        if prob > 0.5:
            return 1.0
        else:
            return 0.0

    def logmle(self):
        m, n = np.shape(self.dataMat)
        temp_mle=0
        for i in range(m):
            temp_mle=temp_mle-self.labelMat[i]*self.weights.transpose()*self.dataMat[i].transpose()+np.log(1+np.exp(1+self.weights.transpose()*self.dataMat[i].transpose()))
        return temp_mle

    def NewtonAscent(self):
        m, n = np.shape(self.dataMat)
        m1,n1=np.shape(self.weights)
        if m1*n1==0:
            self.weights=np.zeros((n,1))
            self.weights=np.mat(self.weights)
        h=np.zeros((m,1))
        h=np.mat(h)
        deta1 =np.zeros((n,1))
        deta1=np.mat(deta1)
        deta2 = np.zeros((n, n))
        deta2=np.mat(deta2)
        for i in range(m):
            h[i]=sigmoid(self.weights.transpose()*self.dataMat[i].transpose())
        for i in range(m):
            deta1 = deta1 - self.dataMat[i].transpose() * float(self.labelMat[i] - h[i])
            deta2 = deta2 + float(h[i] * (1 - h[i]))*self.dataMat[i].transpose() * self.dataMat[i]
        self.weights=self.weights-deta2.getI()*deta1;

myclassfier=logRegressClassifier();
dataMat=list();
labelMat=list();
fr = open("watermelon_dataset3.0.txt")
# print fr.readlines()
for line in fr.readlines():
    lineArr = line.strip().split()
    dataLine = [1.0]
    for i in lineArr:
        dataLine.append(float(i))

    label = dataLine.pop()
    # print dataLine
    dataMat.append(dataLine)
    labelMat.append(int(label))
# help(dataLine)
myclassfier.dataMat = np.mat(dataMat);
myclassfier.labelMat = np.mat(labelMat).transpose();
print np.shape(myclassfier.dataMat)
print np.shape(myclassfier.labelMat)
myclassfier.train()