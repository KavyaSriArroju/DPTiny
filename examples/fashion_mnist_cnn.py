"""Train a CNN on Fashion-MNIST."""

import time
import numpy as np

from dptiny import (
    Variable,
    is_available,
    is_gpu,
    no_grad,
    softmax_cross_entropy,
    test_mode,
    to_gpu,
    use_gpu,
)
from dptiny.backend import xp
from dptiny.data import DataLoader
from dptiny.data.fashion_mnist import get_fashion_mnist, CLASSES
from dptiny.nn import (
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)
from dptiny.optim import Adam


# Random seed for reproducibility.
SEED = 42
np.random.seed(SEED)
xp.random.seed(SEED)


if is_available():
    use_gpu()
    print("GPU enabled for training.")
else:
    print("GPU not available; training on CPU.")


print("Loading Fashion-MNIST dataset...")

X_train, X_test, y_train, y_test = get_fashion_mnist(
    normalize=True,
    flatten=False,
)


# Standardize using ONLY the training-set statistics.
train_mean = float(X_train.mean())
train_std = float(X_train.std())

X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

print(f"Training mean: {train_mean:.6f}")
print(f"Training std: {train_std:.6f}")


if is_gpu():
    X_train = to_gpu(X_train)
    X_test = to_gpu(X_test)
    y_train = to_gpu(y_train)
    y_test = to_gpu(y_test)


# Same CNN architecture as examples/mnist_cnn.py.
model = Sequential(
    Conv2d(1, 16, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Conv2d(16, 32, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Flatten(),
    Linear(32 * 7 * 7, 128),
    ReLU(),
    Dropout(0.3),
    Linear(128, 10),
)

if is_gpu():
    model.to_gpu()


batch_size = 64
max_epoch = 10

data_loader = DataLoader(
    (X_train, y_train),
    batch_size
)

test_loader = DataLoader(
    (X_test, y_test),
    batch_size,
    shuffle=False
)

optimizer = Adam(model, lr=0.001)

start_time = time.time()

for epoch in range(max_epoch):
    # -------------------------
    # Training
    # -------------------------
    sum_loss = 0.0
    train_correct = 0
    train_count = 0

    model.train()

    for x, t in data_loader:
        x = Variable(x)

        y = model(x)
        loss = softmax_cross_entropy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        batch_size_actual = len(t)

        sum_loss += float(loss.data) * batch_size_actual

        pred = y.data.argmax(axis=1)
        train_correct += int((pred == t).sum())
        train_count += batch_size_actual

    train_loss = sum_loss / train_count
    train_acc = train_correct / train_count

    # -------------------------
    # Testing
    # -------------------------
    test_correct = 0
    test_count = 0

    with test_mode(), no_grad():
        for x, t in test_loader:
            y = model(Variable(x))
            pred = y.data.argmax(axis=1)

            test_correct += int((pred == t).sum())
            test_count += len(t)

    test_acc = test_correct / test_count

    print(
        f"Epoch {epoch + 1:2d}/{max_epoch} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Test Acc: {test_acc:.4f} | "
        f"Time: {time.time() - start_time:.2f}s"
    )


# -------------------------------------------------
# Final evaluation and confusion matrix
# -------------------------------------------------

confusion_matrix = np.zeros((10, 10), dtype=np.int64)

with test_mode(), no_grad():
    for x, t in test_loader:
        y = model(Variable(x))
        pred = y.data.argmax(axis=1)

        # Convert GPU arrays to NumPy if necessary.
        if is_gpu():
            pred_np = pred.get()
            t_np = t.get()
        else:
            pred_np = np.asarray(pred)
            t_np = np.asarray(t)

        for true_label, predicted_label in zip(t_np, pred_np):
            confusion_matrix[int(true_label), int(predicted_label)] += 1


print("\n10x10 Confusion Matrix:")
print(confusion_matrix)


# -------------------------------------------------
# Accuracy for each class
# -------------------------------------------------

print("\nPer-class Accuracy:")

for i, class_name in enumerate(CLASSES):
    total = confusion_matrix[i].sum()
    correct = confusion_matrix[i, i]

    class_accuracy = correct / total

    print(
        f"{i} {class_name:12s}: "
        f"{class_accuracy:.4f} "
        f"({correct}/{total})"
    )


# -------------------------------------------------
# Find the pair of classes confused most often
# -------------------------------------------------

pair_confusions = confusion_matrix + confusion_matrix.T

# Ignore the diagonal because it contains correct predictions.
np.fill_diagonal(pair_confusions, 0)

class_a, class_b = np.unravel_index(
    np.argmax(pair_confusions),
    pair_confusions.shape,
)

print("\nMost frequently confused pair:")
print(
    f"{CLASSES[class_a]} <-> {CLASSES[class_b]} "
    f"({pair_confusions[class_a, class_b]} total confusions)"
)

print(
    f"\nTraining completed in "
    f"{time.time() - start_time:.2f} seconds"
)
