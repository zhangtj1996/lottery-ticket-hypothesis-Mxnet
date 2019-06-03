# lottery-ticket-hypothesis-Mxnet
A reimplementation of "The Lottery Ticket Hypothesis" (Frankle and Carbin) by Mxnet for FC network. Some of the code are directly borrowed from the first reference.



If you are using mxnet with CPU version, please modify the `ctx`. 
This implement is for MLP with `relu` activation, for regression problem. If you want to do classification tasks, please modify the loss function and forward function in `Class MaskedNet`.


## Requirment
- numpy 
- pandas
- sklearn
- mxnet >=1.3.0 (GPU version)
- matplotlib

## Reference
- https://github.com/google-research/lottery-ticket-hypothesis
- The Lottery Ticket Hypothesis: Finding Small, Trainable Neural Networks [https://arxiv.org/abs/1803.03635](https://arxiv.org/abs/1803.03635)
