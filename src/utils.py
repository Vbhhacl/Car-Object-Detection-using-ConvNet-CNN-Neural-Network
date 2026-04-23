import matplotlib.pyplot as plt

def plot_results(results):
    for name, history in results.items():
        plt.plot(history.history['val_accuracy'], label=name)

    plt.title("Optimizer Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("results/comparison.png")
    plt.show()