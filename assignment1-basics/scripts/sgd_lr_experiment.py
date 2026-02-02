def run_sgd(lr, iters=10):
    w = 0.0
    losses = []
    for i in range(iters):
        loss = (w - 3.0) ** 2
        grad = 2 * (w - 3.0)
        w = w - lr * grad
        losses.append(loss)
    return losses, w


if __name__ == "__main__":
    lrs = [1e1, 1e2, 1e3]
    for lr in lrs:
        losses, w = run_sgd(lr, iters=10)
        print(f"lr={lr}: final w={w:.6e}, losses={', '.join(f'{x:.6e}' for x in losses)}")
