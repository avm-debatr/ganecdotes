# Hyper-parameters for one-shot segmentor (NOT the clustering model)
lr = 0.001
beta1, beta2 = 0.9, 0.99
weight_decay = 0.000

losses = ['cross_entropy']
lambdas = [1.]

scheduler_type = 'step'
scheduler_args = dict(step_size=500,
                      gamma=0.1)

num_epochs  = 200
print_freq  = 10