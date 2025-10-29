import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_log.csv")
plt.plot(df["epoch"], df["train_acc"], label="Train Accuracy")
plt.plot(df["epoch"], df["test_acc"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_vs_epoch.png", dpi=200)
plt.show()
