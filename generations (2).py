from Global_parameters import *
import numpy as np
channel_train = np.load('channel_train.npy')
train_size = channel_train.shape[0]
channel_test = np.load('channel_test.npy')
test_size = channel_test.shape[0]


def training_gen(bs, SNRdb=20):
    while True:
        index = np.random.choice(np.arange(train_size), size=bs)
        H_total = channel_train[index]
        input_samples = []
        input_labels = []
        for H in H_total:
            bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
            signal_output, para = ofdm_simulate(bits, H, SNRdb)
            #input_labels.append(bits[0:16])
            for i in range(int(len(bits)/2)):
                if bits[2*i]==0 and bits[2*i+1]==0:
                    input_labels.append(0)
                elif bits[2*i]==0 and bits[2*i+1]==1:
                    input_labels.append(1)
                elif bits[2*i] == 1 and bits[2*i+1] == 0:
                    input_labels.append(2)
                elif bits[2*i] == 1 and bits[2*i+1] == 1:
                    input_labels.append(3)
            input_samples.append(signal_output)
            #print(len(input_labels))
        yield (np.asarray(input_samples), np.asarray(input_labels))


def validation_gen(bs, SNRdb=20):
    while True:
        index = np.random.choice(np.arange(test_size), size=bs)
        H_total = channel_test[index]
        input_samples = []
        input_labels = []
        for H in H_total:
            bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
            signal_output, para = ofdm_simulate(bits, H, SNRdb)
            # input_labels.append(bits[0:16])
            for i in range(int(len(bits) / 2)):
                if bits[2 * i] == 0 and bits[2 * i + 1] == 0:
                    input_labels.append(0)
                elif bits[2 * i] == 0 and bits[2 * i + 1] == 1:
                    input_labels.append(1)
                elif bits[2 * i] == 1 and bits[2 * i + 1] == 0:
                    input_labels.append(2)
                elif bits[2 * i] == 1 and bits[2 * i + 1] == 1:
                    input_labels.append(3)
            input_samples.append(signal_output)
        yield (np.asarray(input_samples), np.asarray(input_labels))



        '''
        index = np.random.choice(np.arange(train_size), size=bs)
        H_total = channel_train[index]
        input_samples = []
        input_labels = []
        for H in H_total:
            bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
            signal_output, para = ofdm_simulate(bits, H, SNRdb)
            input_labels.append(bits[0:16])
            input_samples.append(signal_output)
        yield (np.asarray(input_samples), np.asarray(input_labels))
        '''


if __name__ == "__main__":
    x = training_gen(1000, 25)
    print(x)


