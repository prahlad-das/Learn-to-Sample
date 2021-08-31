# Learn to Sample
Framework: Py Geometric: https://github.com/rusty1s/pytorch_geometric <br/>
Concept: GNN

Goal: Learn the sampling distribution used for node sampling in order to improve model accuracy. 

Notation:
- P(n): Probability that a node n in Neighbors is sampled. For example: 
   - Neighborhood sampling use a uniform sampling distribution: p(n) = 1 / |Neighbors| 
   - Neighborhood sampling using importance smapling distribution p(n) = degree(n) / sum_degree(Neighbors)
   - Layerwise importance sampling: p(n) = degree(n) / sum_degree(all nodes)
- D: graph dataset
- GNN(X|W): the GNN model that takes X as inputs with parameters W. 
- Y: ground truth
- Loss(): loss function

Limitations of existing approaches:
- The sampling distribution is manually designed
- The sampling distribution might not fit the graph in hand
- Most of the sampling approaches consider only one node degree information in sampling. However, it is not easy to generalize to heterogeneous graphs. 

Basic idea:
- Apply meta-learning approaches(e.g. black-box optimization or Bayesian optimization) to learn a sampling function
    - Parameterized sampling function p(n|theta)
    - Optimization: 
        - W, Theta = arg min Loss(Y, GNN(X|W, p(n|theta)))
- The first one to purpose the learn to sample framework
- The design of different sampling function p(n)
    - p(n) = a * degree(n) + b * diversity(n) + c * other_metric(n)
    - Use manual designed features to get the other metrics:
    https://web.stanford.edu/class/cs224w/slides/02-tradition-ml.pdf 
        - node degree
        - node centrality
        - clustering coefficient
        - graphlets
- By directly fitting the sampling function to a task/graph, we could get a higher accuracy for the task.
- It is worthwhile to study whether the learned sampling function could easily transfer to similar tasks.
- Provide a programming framework to allow programmers to explore various customized sampling functions to learn (similar to nextdoor paper)
