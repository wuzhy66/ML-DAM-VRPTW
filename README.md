# Meta-Learning-based Deep Reinforcement Learning for Multiobjective Optimization Problems (VRPTW)

## Dependencies

* Python>=3.6
* NumPy
* [PyTorch](http://pytorch.org/)>=1.4

## Meta-Learning

For training meta-model on MOCVRPTW-50 instances:

```bash
python run.py --graph_size 50 --CUDA_VISIBLE_ID "0" --is_train --meta_iterations 2000
```

## Fine-tuning

For fine-tuning the trained meta-model (ML-DAM) on MOCVRPTW-50 instances with 10-step per subproblem:
```bash
python run.py --graph_size 50 --is_load --load_path "meta-model-VRPTW50.pt" --CUDA_VISIBLE_ID "0" --is_test --update_step_test 10
```

For fine-tuning the random-model (DAM) on MOCVRPTW-50 instances with 10-step per subproblem:

```bash
python run.py --graph_size 50 --CUDA_VISIBLE_ID "0" --is_test --update_step_test 10
```

## Transfer-Learning

For training all the submodels with transfer-learning by loading the well trained 1st-submodel (DAM(transfer-obj2)) on MOCVRPTW-50 instances with 10-step per subproblem:

```bash
python run.py --graph_size 50 --is_load --load_path "model-obj2.pt" --CUDA_VISIBLE_ID "0" --is_transfer --is_test --update_step_test 10
```

For training all the submodels with transfer-learning by loading the well trained 100th-submodel (DAM(transfer-obj1)) on MOCVRPTW-50 instances with 10-step per subproblem:

```bash
python run.py --graph_size 50 --is_load --load_path "model-obj1.pt" --CUDA_VISIBLE_ID "0" --is_transfer --is_test --update_step_test 10
```

## Acknowledgements

Thanks to [wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route) for getting me started with the code for the Attention Model.

