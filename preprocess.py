from sklearn import preprocessing

#z-score mean=0 std=1
X_scaled = preprocessing.scale(X)

#save mean and std
scaler = preprocessing.StandardScaler().fit(X)
scaler.transform(X)
# 可以直接使用训练集对测试集数据进行转换
scaler.transform([[-1., 1., 0.]])

#缩放到（0，1）之间
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_test_minmax = min_max_scaler.transform(X_test)

min_max_scaler.scale_
min_max_scaler.min_

#Normalization，归一化p范数
X_normalized = preprocessing.normalize(X, norm='l2')

normalizer = preprocessing.Normalizer().fit(X)  # fit does nothing
normalizer.transform(X)  
normalizer.transform([[-1., 1., 0.]])