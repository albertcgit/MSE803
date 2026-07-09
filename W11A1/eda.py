import seaborn as sns
import matplotlib.pyplot as plt

def plot_target(y):
    sns.histplot(y, kde=True)
    plt.title("Target Distribution")
    plt.show()