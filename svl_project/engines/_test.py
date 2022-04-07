import numpy as np

pred = np.random.randint(0, 4, (1, 5, 4))
print(f"pred {pred}")
gt = np.zeros([1, 5, 4])
gt[0][1] = pred[0][1]
gt[0][2] = pred[0][1]
print(f"gt {gt}")
cor = np.equal(pred, gt)
print(f"cor {cor}")
cor = np.equal(np.sum(cor, axis=2), 4)
print(f"em {cor.shape} {cor.sum()}")
total = cor.shape[1]
print(f"total {total}")