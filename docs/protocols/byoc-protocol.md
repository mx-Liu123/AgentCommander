# BYOC Protocol (Bring Your Own Code)

To integrate your existing training code with AgentCommander, follow this contract.

## 1. Interface
Your `strategy.py` must save weights and be loadable by the evaluator.

```python
# strategy.py
def load_trained_model(path, device):
    model = MyModel()
    model.load_state_dict(torch.load(path))
    return model
```

## 2. Evaluation
Your `evaluator.py` must print exactly: `Best metric: 0.123` (Lower case 'm').

## 3. Data Safety
Do not modify test data in memory. The system includes anti-cheating checks that will invalidate your score if you do.
