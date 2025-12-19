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
