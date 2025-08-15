# Transformer-CNN_PPG-to-ECG-and-ABP


This project predicts **ECG** and **ABP** waveforms from a **PPG** signal using a shared Transformer encoder and two CNN decoders.

## Dataset
- **PulseDB**: [Link](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2022.1090854/full)

## Model Overview
- **Shared Transformer Encoder** processes the PPG signal.
- **Two CNN Decoders**:  
  - One for ECG output  
  - One for ABP output  
- **Weighted Parameter**: Learns the balance between decoders during training.
- **Loss Functions**:  
  - **ABP**: Mean Squared Error (MSE)  
  - **ECG**: Weighted loss (R-peak locations weighted ×10 for better peak capture)

## Training
- `Train_model.py` handles dataset loading, preprocessing, model compilation, and training.
- Model architecture is defined separately for modularity.

---



### Proposed Model
<img width="1272" height="627" alt="image" src="https://github.com/user-attachments/assets/8b9f91bc-58f2-4077-a327-0757e082a8c8" />

### CNN Decoder
<img width="511" height="478" alt="image" src="https://github.com/user-attachments/assets/7b3af5a1-0b12-4061-ba87-99b04842f665" />


