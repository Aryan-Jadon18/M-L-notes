# 🔄 Recurrent Neural Networks (RNNs) – Beginner Tutorial

Recurrent Neural Networks (RNNs) are a class of neural networks designed to handle **sequential data** such as text, speech, time series, and video frames. Unlike feedforward networks, RNNs have **memory**: they use information from previous steps to influence the current output.

---

## 📌 Key Concepts

- **Sequential Data**: RNNs are ideal for data where order matters (e.g., sentences, stock prices).
- **Hidden State**: At each time step, RNNs maintain a hidden state that captures information from previous inputs.
- **Weights Sharing**: The same weights are applied across all time steps, making RNNs efficient for sequences.
- **Backpropagation Through Time (BPTT)**: Training method that unfolds the RNN across time steps to compute gradients.

---

## 🧠 RNN Architecture (Simplified)

Imagine a loop:

Input (x_t) → Hidden State (h_t) → Output (y_t)
↑___________________________↓


- At each time step `t`:
  - `h_t = f(W_x * x_t + W_h * h_(t-1))`
  - `y_t = g(W_y * h_t)`

Where:
- `x_t` = input at time step `t`
- `h_t` = hidden state
- `y_t` = output
- `W_x, W_h, W_y` = weight matrices
- `f, g` = activation functions (e.g., tanh, softmax)

---

## ⚠️ Challenges
- **Vanishing/Exploding Gradients**: RNNs struggle with long sequences.
- **Solutions**: LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) are advanced variants that solve these issues.

---

## 🛠️ Typical Code Example (PyTorch)

```python
import torch
import torch.nn as nn

# Sample RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, hidden = self.rnn(x)  # hidden: last hidden state
        out = self.fc(out[:, -1, :])  # use last time step output
        return out

# Example usage
input_size = 10   # features per time step
hidden_size = 20  # hidden neurons
output_size = 2   # binary classification

model = SimpleRNN(input_size, hidden_size, output_size)

# Dummy input: batch of 5 sequences, each of length 7, with 10 features
x = torch.randn(5, 7, 10)
output = model(x)

print("Output shape:", output.shape)  # (5, 2)
```
---
### 📈 Applications
Natural Language Processing (NLP): Text generation, sentiment analysis, machine translation.

Speech Recognition: Converting audio signals into text.

Time Series Forecasting: Stock prices, weather prediction.

Video Analysis: Frame-by-frame sequence modeling.

---

### ✅ Summary
RNNs are powerful for sequential data but limited by gradient issues.

LSTM and GRU are improved versions widely used in practice.

PyTorch makes it easy to implement RNNs with nn.RNN, nn.LSTM, and nn.GRU.

---

