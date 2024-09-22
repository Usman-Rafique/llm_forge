# GPT-2 Medium Training Log

## Configuration
- Model: GPT-2 Medium
- Training Duration: 24 hours
- Batch Size: 4
- Learning Rate: 1e-4
- Gradient Clipping: 1.0

## Training Progress

### Training Loss
![Training Loss](./images/gpt2_medium_train_loss.png)

### Validation Loss
![Validation Loss](./images/gpt2_medium_val_loss.png)

## Evaluation Results
Final validation loss: 3.38

## Sample Output
```
Fruits are good for you because fruits contain vitamin C and vitamin E.
Fruits also contain fiber, which is very important for the health of the brain and the mind.
According to the Food and Agriculture Organization (FAO) it is recommended to consume a wide range of fruits and vegetables for the health of your body and the health of your brain and the mind.
Fruits are rich in vitamins, fiber, and vitamins and are also rich in vitamins and minerals.
Fruits are rich in vitamins and fiber as
```

## Challenges and Solutions
- Issue: NaN loss
  Solution: Implemented gradient clipping with a max norm of 1.0
- Issue: Special token '<|endoftext|>' in training data
  Solution: Added `allowed_special` in the tokenization to fix it: `tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})`
- Issue: Learning rate of `3e-4` seemed fine, but later found out that the learning rate should be `1e-4` led to faster convergence. I still have not tried even lower learning rates or learning rate schedules.
- Issue: Mixed Precision did not reduce GPU memory usage
  - I have not found a way to reduce GPU memory usage using mixed precision. My setup is 2 x RTX 4090 GPUs with 24GB RAM each. I have not yet tried gradient accumulation.