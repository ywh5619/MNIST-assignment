# MNIST-assignment
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
