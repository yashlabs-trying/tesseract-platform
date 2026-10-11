# v4: Distributed Inference

**Goal:** Scale inference across multiple nodes with intelligent sharding and routing.

### Features
- Tensor parallelism across GPUs/nodes
- Pipeline parallelism
- Request-level load balancing
- Model sharding (expert parallelism)
- Auto-scaling worker pools
- Fault tolerance & retry

### Structure
```
v4-distributed/
├── src/
│   ├── parallel/     # Parallelism strategies
│   ├── shard/        # Model sharding
│   ├── balancer/     # Load balancer
│   └── cluster/      # Cluster management
├── tests/
└── requirements.txt
```
