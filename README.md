# MNIST-assignment
layers_1.py和mnist_mlp_cpu.py是代码部分
运行mnist_mlp_cpu.py，其中mnist_data是就是手写数据集
训练部分包含显示了总轮数、总训练时间，训练完成后的模型的文件为mlp-256-128-64-30epoch.npy上传到了百度网盘中（链接在实验报告的pdf中）
代码中的RelU对应的是要求中的L2正则化，训练的过程就是一个SGD的流程
用def load_model(self, param_dir)重新导入模型,反向传播和正向传播设定是三层，out_classes决定了这个是10分类问题（手写数据集的识别)，lr是模型的learning rate
部分训练和测试部分的代码如下：
def build_mnist_mlp(param_dir='weight.npy'):
    h1, h2, h3, e = 256, 128 ,64, 30
    mlp = MNIST_MLP(hidden1=h1, hidden2=h2, hidden3=h3, max_epoch=e)
    mlp.load_data()
    mlp.build_model()
    mlp.init_model()
    start_time = time.time()
    mlp.train()
    print("All train time: %f"%(time.time()-start_time))
    mlp.save_model('mlp-%d-%d-%d-%depoch.npy' % (h1, h2, h3, e))
    # mlp.load_model('mlp-%d-%d-%depoch.npy' % (h1, h2, e))
    return mlp

if __name__ == '__main__':
    mlp = build_mnist_mlp()
    mlp.evaluate()
