# LSTM for beginners

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) designed to learn from sequential data by selectively remembering and forgetting information over long time spans. They’re widely used in NLP, speech, and time-series forecasting.

---

## Introduction to LSTM

- **Purpose:** Handle sequences where past context matters (e.g., sentences, sensor readings).
- **Problem solved:** Mitigates vanishing/exploding gradients seen in vanilla RNNs during backpropagation through time.
- **Core idea:** A memory cell carries long-term information; gates control what to keep, add, and output.

> Tip: Think of the cell state as a conveyor belt; gates act like valves controlling the flow of information.

---

## Core concepts and math

- **Cell state:** Long-term memory flowing across time steps.
- **Hidden state:** Short-term context used for outputs at each step.
- **Gates:** Sigmoid-controlled filters:
  - **Forget gate:** What to discard.
  - **Input gate:** What new info to add.
  - **Output gate:** What to expose to the next layer/output.

- **Typical gate equations:**
  
  

\[
  f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
  \]


  

\[
  i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
  \]


  

\[
  \tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)
  \]


  

\[
  c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
  \]


  

\[
  o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
  \]


  

\[
  h_t = o_t \odot \tanh(c_t)
  \]



---

## Typical code structure (PyTorch)

```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        direction_multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_multiplier, output_size)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        out, (hn, cn) = self.lstm(x)             # out: [batch, seq_len, hidden*dirs]
        last_step = out[:, -1, :]                # take last time step
        logits = self.fc(last_step)              # map to output
        return logits

# Example usage
batch_size, seq_len, input_size = 32, 50, 16
hidden_size, output_size = 64, 1

model = LSTMModel(input_size, hidden_size, output_size, num_layers=2, bidirectional=True, dropout=0.2)
x = torch.randn(batch_size, seq_len, input_size)
y_pred = model(x)
print(y_pred.shape)  # torch.Size([32, 1])
```
## 🔄 How It Works Step-by-Step

Input sequence enters one step at a time.

Forget gate decides what past info to drop.

Input gate updates memory with new info.

Cell state carries long-term memory forward.

Output gate produces hidden state for predictions.

Training uses backpropagation through time (BPTT), but LSTM’s design prevents gradient vanishing.
---
## 📌 Practical Concepts You Must Know

Sequence length: How many time steps per sample.

Batching: Group sequences for faster training.

Bidirectional LSTMs: Process sequences forward and backward.

Stacked LSTMs: Multiple layers for deeper learning.

Dropout: Prevents overfitting.

Packed sequences: Handle variable-length sequences efficiently.


---
## 🔍 When to Use LSTM vs Alternatives

| Model        | Best for                           | Pros                          | Cons                          |
|--------------|------------------------------------|-------------------------------|-------------------------------|
| **LSTM**     | Medium-length sequences, limited data | Handles long dependencies      | Slower, complex to tune       |
| **GRU**      | Similar tasks, simpler models      | Fewer parameters, faster       | Slightly less expressive      |
| **Transformer** | Long-range context, large datasets | Parallelizable, SOTA in NLP    | Data-hungry, more compute     |


##⚠️ Common Pitfalls and Fixes

Overfitting: Use dropout, reduce layers, add regularization.

Exploding gradients: Clip gradients (torch.nn.utils.clip_grad_norm_).

Tensor shape errors: Ensure [batch, seq_len, features] when batch_first=True.

Variable lengths: Use packed sequences or pad consistently.

Wrong outputs: For classification, use last time step; for seq2seq, keep all steps.

---
## 🐍 Minimal TensorFlow/Keras Example

import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(50, 16)),                 # seq_len, features
    layers.LSTM(64, return_sequences=False),      # set True if you need all steps
    layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")
x = tf.random.normal((32, 50, 16))
y = tf.random.normal((32, 1))
model.fit(x, y, epochs=3, batch_size=32)

Tune hyperparameters: hidden size, layers, bidirectional, dropout.

Evaluate with correct metrics (MSE, accuracy, F1).

Compare against baselines (GRU, MLP).


