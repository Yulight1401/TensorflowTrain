# Tensorflow & Machine Learning

---

This res is used to storage the demos I made, when I studying ML

---

AdaBoost:


``` shell
D:\GitAPP\TensorflowTrain>python adaboost.py
dataMat: [[ 1.   2.1]
 [ 2.   1.1]
 [ 1.3  1.1]
 [ 1.   1.1]
 [ 2.   1. ]] classLabels: [1.0, 1.0, -1.0, -1.0, 1.0]
D: [[ 0.2  0.2  0.2  0.2  0.2]]
classEst： [[-1.  1. -1. -1.  1.]]
aggClassEst: [[-0.49314718  0.89314718 -0.49314718 -0.49314718  0.89314718]]
total error:  0.2
D: [[ 0.5    0.125  0.125  0.125  0.125]]
classEst： [[ 1.  1.  1.  1.  1.]]
aggClassEst: [[ 0.05615896  1.44245332  0.05615896  0.05615896  1.44245332]]
total error:  0.4
D: [[ 0.33333333  0.08333333  0.25        0.25        0.08333333]]
classEst： [[ 1. -1. -1. -1. -1.]]
aggClassEst: [[ 0.86087792  0.63773437 -0.74855999 -0.74855999  0.63773437]]
total error:  0.0
classifierArray: [{'alpha': 0.6931471805599453, 'dim': 0, 'thresh': 1.3, 'ineq': 'lt'}, {'alpha': 0.5493061443340548, 'dim': 0, 'thresh': 0.90000000000000002, 'ineq': 'lt'}, {'alpha': 0.8047189562170503, 'dim': 1, 'thresh': 1.1100000000000001, 'ineq': 'lt'}]
[[-0.69314718]] 0.6931471805599453 [[-1.]]
[[-0.14384104]] 0.5493061443340548 [[ 1.]]
[[ 0.66087792]] 0.8047189562170503 [[ 1.]]
result: [[ 1.]]

```
