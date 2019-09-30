from mxnet import nd


class Equation():
    def __init__(self, max_num=5, output_len=5, padding_len=4, ctx=None, *args):
        super(Equation, self).__init__(*args)
        self.idx2str = [0,1,2,3,4,5,6,7,8,9,'+','<bos>','<pad>','<eos>']
        self.str2idx = {'+':10, '<bos>':11, '<pad>':12, '<eos>':13}
        # self.str2idx = {'+':5, '<bos>':6, '<pad>':7, '<eos>':8}
        self.test_set = []
        self.ctx = ctx
        for i in range(max_num):
            for j in range(max_num):
                equation = [i,self.str2idx['+'],j]
                result = [int(t) for t in str(i+j)]
                result.append(self.str2idx['<eos>'])
                result = [self.str2idx['<bos>']] + result
                while len(result) < padding_len:
                    result = result + [self.str2idx['<pad>']]
                self.test_set.append((equation, result))
    
    def generate_data_iter(self, batch_size):
        test_set_size = len(self.test_set)
        batch_num = test_set_size // batch_size
        for i in range(batch_num):
            X_raw = [pair[0] for pair in self.test_set[i*batch_size: i*batch_size+batch_size]]
            Y_raw = [pair[1] for pair in self.test_set[i*batch_size: i*batch_size+batch_size]]
            # X = nd.array([[self.str2idx[s] for s in equ] for equ in X_raw], ctx=self.ctx)
            # Y = nd.array([[self.str2idx[s] for s in equ] for equ in Y_raw], ctx=self.ctx)
            X = nd.array(X_raw, ctx=self.ctx)
            Y = nd.array(Y_raw, ctx=self.ctx)
            yield X, Y
    

if __name__=='__main__':
    equ = Equation(10)
    for x,y in equ.generate_data_iter(5):
        print(x)
        print(y)
