# utils.py
import matplotlib.pyplot as plt

def plot_loss(loss_history, save_path='loss_curve.png'):
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(save_path)
    plt.show()
