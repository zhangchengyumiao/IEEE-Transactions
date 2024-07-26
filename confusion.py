class ClassifierMixin:
    """Mixin class for all classifiers in scikit-learn."""
    _estimator_type = "classifier"

    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.

        """
        from .metrics import accuracy_score
        import matplotlib.pyplot as pl
        from sklearn import metrics
        ##############################################################################################################################################
        if accuracy_score(y, self.predict(X), sample_weight=sample_weight)>0.01:



         def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=1.0, axis_labels=None):
            # 利用sklearn中的函数生成混淆矩阵并归一化
            cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

            # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
            pl.imshow(cm, interpolation='nearest', cmap=pl.get_cmap('Blues'))
            pl.colorbar()  # 绘制图例

            # 图像标题
            if title is not None:
                pl.title(title)
            # 绘制坐标
            num_local = np.array(range(len(labels_name)))
            if axis_labels is None:
                axis_labels = labels_name
            pl.xticks(num_local, axis_labels, rotation=0)  # 将标签印在x轴坐标上， 并倾斜0度
            pl.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
            pl.ylabel('True label')
            pl.xlabel('Predicted label')
            pl.show()

         plot_matrix(y, self.predict(X), [k for k in range(0,120)], title=None, thresh=1.0, axis_labels=None)
         #####################################################################################################################################################
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)