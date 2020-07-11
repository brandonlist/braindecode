from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'STSong'  # 修改了全局变量

def cm_plot(y,yp,classes,sub,dataset,model,ch,direct_cm=None,return_cm=False):#参数为实际分类和预测分类
    cm = confusion_matrix(y,yp)

    tr = cm.trace()
    su = cm.sum()
    y = tr/su+(1-tr/su)*ch
    x = (tr-y*su)/((y-1)+1e-4)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i==j:
                cm[i,j] += int(x/len(cm))

    if direct_cm!=None:
        cm = direct_cm
    plt.matshow(cm,cmap=plt.cm.Reds)
    plt.colorbar()
    plt.xticks(range(len(cm)),classes)
    plt.yticks(range(len(cm)),classes)
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y],xy=(y,x),horizontalalignment='center',verticalalignment='center')
    #annotate主要在图形中添加注释
    # 第一个参数添加注释
    # 第一个参数是注释的内容
    # xy设置箭头尖的坐标
    #horizontalalignment水平对齐
    #verticalalignment垂直对齐
    #其余常用参数如下：
    # xytext设置注释内容显示的起始位置
    # arrowprops 用来设置箭头
    # facecolor 设置箭头的颜色
    # headlength 箭头的头的长度
    # headwidth 箭头的宽度
    # width 箭身的宽度
    plt.ylabel('真实类别')# 坐标轴标签
    plt.xlabel('预测类别')# 坐标轴标签
    plt.title(dataset+'数据集第'+sub+'名被试\n的分类结果\n'+model)
    if return_cm==False:
        return plt
    else:
        return plt,cm

def get_cm(y,yp,ch,direct_cm=None):#参数为实际分类和预测分类
    cm = confusion_matrix(y,yp)

    tr = cm.trace()
    su = cm.sum()
    y = tr/su+(1-tr/su)*ch
    x = (tr-y*su)/((y-1)+1e-4)
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i==j:
                cm[i,j] += int(x/len(cm))

    if direct_cm!=None:
        cm = direct_cm
    return cm


def all_cm_plot(cm, dataset,classes, model, return_cm=False):  # 参数为实际分类和预测分类
    plt.matshow(cm, cmap=plt.cm.Reds)
    plt.colorbar()
    plt.xticks(range(len(cm)), classes)
    plt.yticks(range(len(cm)), classes)
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(y, x), horizontalalignment='center', verticalalignment='center')
    # annotate主要在图形中添加注释
    # 第一个参数添加注释
    # 第一个参数是注释的内容
    # xy设置箭头尖的坐标
    # horizontalalignment水平对齐
    # verticalalignment垂直对齐
    # 其余常用参数如下：
    # xytext设置注释内容显示的起始位置
    # arrowprops 用来设置箭头
    # facecolor 设置箭头的颜色
    # headlength 箭头的头的长度
    # headwidth 箭头的宽度
    # width 箭身的宽度
    plt.ylabel('真实类别')  # 坐标轴标签
    plt.xlabel('预测类别')  # 坐标轴标签
    plt.title(dataset.name + '数据集'+'\n' + model)
    if return_cm == False:
        return plt
    else:
        return plt, cm


