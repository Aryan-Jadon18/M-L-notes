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
