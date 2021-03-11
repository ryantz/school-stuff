from tensorflow.python.keras import *
from tensorflow.python.keras.layers import *
from generations import *
import tensorflow as tf
import numpy as np
import argparse

def main():
    n = 4096
    n_t = 500
    train_SNR = 15
    train_epochs = 20
    train = False

    #Ill add this feature later
    '''
    parser = argparse.ArgumentParser(description="Description for the project")
    parser.add_argument("-n", "--trials", help="Number of times to repeat the experiment", type=int, required=False,
                        default=1000)
    parser.add_argument("-n_t", "--trials_for_test", help="Number of times to repeat the test", type=int, required=False,
                        default=500)
    parser.add_argument("-train", help="If to train", type=bool, required=False,
                        default=False)
    '''

    # class_names is how labels are assigned
    class_names = ['00', '01', '10', '11']
    num_classes = 4

    # DNN model with three hidden layer
    model = Sequential([
        Dense(n_hidden_1, activation='relu', input_shape=(payloadBits_per_OFDM * 2,)),
        Dense(n_hidden_2, activation='relu'),
        Dense(n_hidden_3, activation='softmax'),
        Dense(num_classes)
    ])

    # how to train: use cross_entropy
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),)
    # output the model
    model.summary()

    # train the model and validation if needed, save the weights for further use
    if train:
        model.fit(
            training_gen(n, train_SNR),
            epochs=train_epochs,
            steps_per_epoch=500,
            validation_data=validation_gen(n, train_SNR),
            validation_steps=1,
            verbose=2)
        model.save_weights('./15test')

    # load the weights
    model.load_weights('./15test')  # ('./temp_trained_25.h5')
    loss = []
    SNRl = []
    BERl = []

    # test the model under different SNR, we use 5,10,15,20,25,30 here
    for SNR in range(5, 30, 5):


        # ignore index, it is for signal part
        #index = np.random.choice(np.arange(train_size), size=10000)
        # bits_org is to record the original data bits
        global bits_org
        bits_org = []

        test_data = test_gen(13400, SNR)

        test_loss = model.evaluate(
            validation_gen(1000, SNR),
            steps=1
        )
        # test_loss is cross_entropy
        loss.append(test_loss)

        # model.predict will return the output of DNN e.g.[0.1,0.9,0,0]...
        result = model.predict(
            test_data,
            batch_size=None,
            verbose=0,
            steps=1
        )
        bits_re = []
        # restore will fill bits_re with the data we restore from the output
        restore(result, bits_re)

        # error is to record  the difference the data to calculate bit error rate
        error = 0
        for i in range(len(bits_re)):
            if bits_re[i] != bits_org[i]:
                error += 1
        Ber = error / len(bits_re)
        print("SNR", SNR)
        SNRl.append(SNR)
        print("BRE:",Ber)
        BERl.append(Ber)
    # the problem now is that for different SNR the BER are similar

    print("SNR:",SNRl)
    print("loss:",loss)
    print("BER:",BERl)

def test_gen(bs, SNRdb=20):
    global bits_org
    while True:
        index = np.random.choice(np.arange(test_size), size=bs)
        H_total = channel_train[index]
        input_samples = []
        input_labels = []
        for H in H_total:
            bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM))
            bits_org.append(bits[0])
            bits_org.append(bits[1])
            signal_output, para = ofdm_simulate(bits, H, SNRdb)
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

if __name__ == "__main__":
    main()
