# FL_demo 


Federated learning demonstration 


Dependency: PyTorch 1.10.1 


Basic usage: python main.py 


- Runs an FL simulation with 10 clients running FedSGD for 50 rounds on MNIST dataset distributed among the clients in a non-IID manner.



Args: 


  dataset: 'mnist' and 'cifar10' supported 
  
  bias: choose from [0, 1] as the non-IID bias for data distribution among clients, 0 represents IID 

  num_workers: number of clients in the FL system 

  cmax: parameter needed by some aggregation techniques that assumes a maximum number on unreliable clients (~ 20% of num_workers is a good choice) 
 
  num_rounds: number of rounds of FL 
  
  num_iters: number of local iteration steps for FedAvg (num_iters = 1 simulates FedSGD) 

  gpu: enter 0 if GPU is available to accelerate training 
  
  aggregation: global model update rule. See aggregation.py for the list of available options to choose from 
  
  exp: name of the experiment - used as prefix for saved numpy files 

  

Example: 


python main.py --dataset cifar10 --bias 0 --num_workers 20 --cmax 0 --num_rounds 100 --num_iters 5 --gpu 0 --aggregation fedsgd --exp train_cifar 
