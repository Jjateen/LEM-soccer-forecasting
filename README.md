# **Hack-A-Sol 2024-25: AI/ML Track PS-1**

## **Project Overview**
This project is built for **Hack-A-Sol 2024-25**, hosted by IIIT NR. The focus is to apply **Large Events Models (LEMs)** for sequential event prediction in soccer, using the **Wyscout dataset**. LEMs are used to simulate matches, predict player performance, and fine-tune models for various contextual scenarios, such as different teams and player behaviors.

The project includes two sets of notebooks:
1. **Notebooks satisfying the hackathon's problem statement (PS).**
2. **Notebooks inside the "ScaledUP" directory** that represent **upscaled versions**, fine-tuned on more data than required by the problem statement.

---

## **Directory Structure**
```
LargeEventsModel/
│
├── data/
│   ├── wyscout/
│   │   ├── csv/
│   │   │   ├── events/ (CSV Files)
│   │   ├── json/
│   │       ├── events/ (JSON Files)
│   │       ├── matches/ (Match Data)
├── lib/
│   ├── data_utils.py        # Helper functions for handling Wyscout data
│   ├── model_utils.py       # Utility functions for LEMs model building
│   ├── simulator.py         # Simulation functions using LEMs
├── models/
│   ├── lem/
│       ├── LEMv3_MODEL_DATA_TORCH.pth  # Pre-trained LEM data model
│       ├── LEMv3_MODEL_TYPE_TORCH.pth  # Pre-trained LEM type model
│       ├── LEMv4_MODEL_ACC_TORCH.pth   # Pre-trained accuracy model
├── ScaledUP/                # Upscaled versions, fine-tuned on more data than expected by PS
├── Dockerfile               # Docker setup for environment consistency
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## **Key Files**

### **1. Notebooks for Hackathon Problem Statement (PS):**
- **`011_Train_Model_Type.ipynb`:** Train the LEM to predict event types.
- **`012_Train_Model_Accuracy.ipynb`:** Train the model to estimate the accuracy of events.
- **`013_Train_Model_Data.ipynb`:** Predict event location and time intervals between events.
- **`021_Learning_State_Values.ipynb`:** Reinforcement learning for learning state values.
- **`022_Valuing_Actions_From_States.ipynb`:** Estimating value of actions, key for decision-making.
- **`023_Finetuning_Framework_Lib.ipynb`:** Fine-tune LEMs based on new data for better predictions.

### **2. Upscaled Notebooks (ScaledUP Directory):**
The notebooks inside **`ScaledUP/`** are designed to handle more data than required by the problem statement, offering more advanced, fine-tuned models:
- **`ScaledUP/011_Train_Model_Type.ipynb`**
- **`ScaledUP/022_Valuing_Actions_From_States.ipynb`**
- **`ScaledUP/013_Train_Model_Data.ipynb`**
  
These notebooks are fine-tuned on larger datasets for more comprehensive insights into event sequences and player performance.

---

## **Technologies and Frameworks**

- **Python** for core development
- **PyTorch** for deep learning model training and inference
- **NumPy** and **Pandas** for data manipulation
- **Jupyter Notebooks** for experimentation and prototyping
- **Docker** for creating reproducible environments

---

## **Setup Instructions**

### **1. Install Dependencies**
Ensure you have Python 3.8+ installed. You can install all necessary dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### **2. Dataset**
The project uses Wyscout soccer data stored in `data/wyscout/csv` and `data/wyscout/json`. The dataset contains detailed match events and metadata for various leagues.

### **3. Running the Models**
- **Event Type Prediction:** Run `011_Train_Model_Type.ipynb` for the basic problem statement or use the **ScaledUP** version for fine-tuning on more data.
- **Accuracy Prediction:** Run `012_Train_Model_Accuracy.ipynb`.
- **Event Data Prediction:** Use `013_Train_Model_Data.ipynb`.

### **4. Simulation**
Simulate matches and predict sequences of game events using **`simulator.py`**, which generates and updates the context based on LEMs’ predictions.

---

## **Acknowledgements**
This project is inspired by Large Language Models (LLMs) and adapted for sequential event prediction in sports, particularly soccer. Special thanks to IIIT NR and Hack-A-Sol organizers for providing a platform to showcase AI in sports analytics.

---

Enjoy the code!
