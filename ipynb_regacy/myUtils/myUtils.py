import h5py
import pandas as pd
import numpy as np
import mmh3
from tqdm import tqdm_notebook as tqdm

class Helper:
    
    def __init__(self):
        self.chunk_no = 0
        self.path = ''

    # h5py to Dataframe
    # 데이터 일부만 사용하고 싶을때 유용
    def makeDF(self, start, end, mode='train'):
        self.h = h5py.File(self.path,'r')

        chunk = self.h[mode] # train, dev
        cols = ['pid', 'product', 'model', 'brand', 'maker', 'price', 'updttm', 'bcateid', 'mcateid', 'scateid', 'dcateid']
        data = {
            c: chunk[c][start : end] for c in cols
        }
        self.df = pd.DataFrame(data)
        self.df['img_feat'] = chunk['img_feat'][start:end]
        
        # utf8 process
        for i in ['pid', 'product', 'brand', 'model', 'maker', 'updttm']:
            self.df[i] = self.df[i].apply(lambda x: x.decode('utf8'))

    # df to data - char to token
    def df2data(self, path, seq_len=100, hash_size=4000):
        data_x = []
        for i in tqdm(list(zip(self.df['product'], self.df['brand'], self.df['maker'], self.df['model']))):
            sentence = ' '.join(i)
            sentence = list(sentence)
            # hash --> word to id
            word_ids = [mmh3.hash(word, seed=2018)%(hash_size) for word in sentence][:seq_len]
            word_ids = np.pad(word_ids, (0,seq_len-len(word_ids)), 'constant', constant_values=(0))
            data_x.append(word_ids)
        
        # save data_x
        np.savetxt(path, data_x, delimiter=',')
        # np.savetxt('tmp/data_x_%d.csv' %(self.chunk_no), data_x, delimiter=',')
        return np.array(data_x)

    # y-label to token
    def cate2token(self, path, y_vocab):
        # cate to token
        data_y=[]
        for cate in tqdm(zip(self.df['bcateid'],self.df['mcateid'],self.df['scateid'],self.df['dcateid'])):
            tmp = str(cate[0])+'>'+str(cate[1])+'>'+str(cate[2])+'>'+str(cate[3])
            data_y.append(y_vocab[tmp])

        # save data_y
        np.savetxt(path, data_y, delimiter=',')
        # np.savetxt('tmp/data_y_%d.csv' %(self.chunk_no), data_y, delimiter=',')
        return np.array(data_y)

    def onehot(self, data_y, output_dim=4215): # 4215개
        # y-label to one-hot encoding
        y_tmp = np.zeros([len(data_y), output_dim], dtype=np.int8)
        y_tmp[np.arange(len(data_y)), data_y] = 1
        data_y = y_tmp

        return data_y

    # 1-8 청크의 상위 0.1%를 validation, 10%를 테스트 데이터로 나눔
    def split_samples(self):
        return

    # inference data만들기
    def make_infer_data(self):
        return



class MyModel():

    def __init__(self):
        return

    def build_model(self):
        return

    def train(self):
        return

    def change_data(self):
        return

    def inference(self):
        return