import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    label_path = "Data/full_labels.npy"
    data_path = "Data/full_data.npy"
    labels = np.load(label_path)
    data = np.load(data_path)

    # sample = data[0]
    # lead = sample[0, 0]
    # df = pd.DataFrame(sample.reshape(-1, sample.shape[-1])).T
    # df.index.set_names(['time'])
    #
    # df.plot(alpha=0.8, legend=False)
    # plt.title("a single sample from the data")
    # plt.show()
    #
    # sns.lineplot(x=np.arange(lead.shape[0]), y=lead)
    # plt.title("a single lead from a sample of the data")
    # plt.show()
    #
    # plt.hist(labels)
    # plt.title("label histogram")
    # plt.show()
    #
    # print("x")


if __name__ == '__main__':
    main()
