# 🚀 AI-Powered Predictive Load Balancer (LSTM–FNN)

A hybrid deep learning–based load balancing system for cloud environments that **predicts future workload** and performs **proactive task scheduling** using an LSTM–FNN architecture.

---

## 📌 Overview

Traditional load balancing algorithms like **Ant Colony Optimization (ACO)** and **Bird Swarm Optimization (BSO)** are reactive — they respond only after overload occurs.

This project introduces a **predictive approach** using:
- 🧠 **LSTM (Long Short-Term Memory)** for time-series forecasting  
- ⚡ **Feedforward Neural Network (FNN)** for intelligent task allocation  

👉 Result: Faster, smarter, and more efficient cloud resource management.

---

## 🎯 Key Features

- 📊 Predicts future VM workload (t+1)
- ⚖️ Intelligent task allocation using fitness function
- 🔄 Proactive VM migration before overload
- 📉 Reduced makespan and response time
- 📈 Improved throughput and resource utilization
- 🧠 Hybrid AI model (LSTM + FNN)

---

## 🧠 Model Architecture

### Input Features
- CPU utilization  
- Memory usage  
- Time-based features  

### Temporal Branch (LSTM)
- LSTM (64 units) → Dropout  
- LSTM (32 units) → Dropout  

### Feature Branch (FNN)
- Dense (ReLU) → BatchNorm → Dropout  
- Dense (ReLU)  

### Fusion Layer
- Concatenation of LSTM + FNN outputs  

### Output
- Predicted VM load  

---

## ⚙️ System Workflow

1. Monitor VM metrics (CPU, memory, queue length)  
2. Predict future workload using LSTM–FNN  
3. Classify VMs (Overloaded / Underloaded)  
4. Compute fitness score:


F = 1 / (makespan × utilization)


5. Allocate task to best VM  
6. Trigger migration if overload is predicted  

---

## 📊 Results

| Metric        | ACO   | BSO   | LSTM–FNN (Proposed) |
|--------------|------|------|--------------------|
| Makespan     | 10.8s | 1.88s | **1.45s** |
| Throughput   | 76%  | 91.55% | **97%** |
| Utilization  | Low  | 55–87% | **65–92%** |

👉 The proposed model significantly outperforms traditional approaches.

---

## 📸 Visualizations

### 🔹 Performance Comparison

### 🔹 Training Graph

### 🔹 Load Balancing

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy / Pandas  
- HTML + Chart.js  
- Git & GitHub  

---

## 🧪 Future Improvements

* ⚡ Reinforcement Learning-based scheduling
* 🌱 Energy-efficient load balancing
* ☁️ Deployment on real cloud platforms (AWS/GCP)
* 📡 Real-time workload streaming

---

## 📄 Research Paper

---

## 🤝 Contributing

Contributions are welcome!
Feel free to open issues or submit pull requests.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Pratyush Raj Srivastava**
B.Tech CSE (AI/ML)
Passionate about AI, Cloud Computing & Real-world Problem Solving 🚀

---



