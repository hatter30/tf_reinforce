
### code

1. policy gradient(REINFORCE)
    1. policy gradient (softmax policy) 
    2. vanilla policy gradient
    
2. actor critic
    1. vanilla actor critic
        1. critic update : episode (monte-carlo)
        2. actor update : with value predicted by critic network
    2. advantage actor critic
        1. critic update : episode (monte_carlo)
        2. actor update : with returns and value predicted by critic network