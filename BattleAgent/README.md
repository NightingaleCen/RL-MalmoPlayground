# BattleAgent

## Training

Training from scratch with 1 zombie.

```bash
python train_dqn.py
```

Training from scratch with multiple mobs.

```bash
python train_dqn.py spider
```



```bash
python train_dqn.py zombie 2
```

Specify the model name.

```bash
python train_dqn.py zombie 1 my_agent
```

The model will be saved in save/my_agent.h5

## Testing

Specify the model to test.

```bash
python begin_arena.py zombie 1 my_agent
```

