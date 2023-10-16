

#usage:
#    TsneSpk(embedings,labels,NumOutput)
# embedings:  numpy.ndarray(n_samples * n_features)
# labels : numpy.ndarray     (n_samples,)
# NumOutput:number of output speakers


#----------------------------------------------
#example
from sklearn import datasets

digits = datasets.load_digits(n_class=9)
embedings= digits.data
labels = digits.target
NumOutput=7
from TsneSpk import TsneSpk
TsneSpk(embedings,labels,NumOutput)
#--------------------------------------------