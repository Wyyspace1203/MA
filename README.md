# Source-Free Model Adaptation for Unsupervised 3D Object Retrieval



## 🌟 Introduction

With the explosive growth of 3D objects yet expensive annotation costs, unsupervised 3D object retrieval has become a popular but challenging research area.
Existing labeled resources have been utilized to aid this task via transfer learning, which aligns the distribution of unlabeled data with the source one.
However, the labeled resource are not always accessible due to the privacy disputes, limited computational capacity and other thorny restrictions.
Therefore, we propose source-free model adaptation task for unsupervised 3D object management, which utilizes a pre-trained model to boost the performance with no access to source data and labels.
Specifically, we compute representative prototypes to assume the source feature distribution, and design a bidirectional cumulative confidence-based adaptation strategy to adaptively align unlabeled samples towards prototypes. Subsequently, a dual-model distillation mechanism is proposed to generate source hypothesis for remedying the absence of ground-truth labels.
The experiments on a cross-domain retrieval benchmark NTU-PSB (PSB-NTU) and a cross-modality retrieval benchmark MI3DOR also demonstrate the superiority of the proposed method even without access to raw data. 

## 📊 Performance

### NTU-PSB
| Method   | NN ↑  | FT ↑  | ST ↑  | F ↑   | DCG ↑  | ANMRR ↓ |
|----------|-------|-------|-------|-------|--------|---------|
| Ours-SF  | 0.7207 | 0.6647 | 0.7652 | 0.4937 | 0.5708 | 0.3289 |
| Ours-SA  | **0.7117** | **0.6962** | **0.7766** | **0.5099** | **0.6664** | **0.3006** |

---

### PSB-NTU
| Method   | NN ↑  | FT ↑  | ST ↑  | F ↑   | DCG ↑  | ANMRR ↓ |
|----------|-------|-------|-------|-------|--------|---------|
| Ours-SF  | 0.8088 | 0.7138 | 0.8123 | 0.5873 | 0.6580 | 0.2754 |
| Ours-SA  | **0.8676** | **0.8074** | **0.8914** | **0.6011** | **0.7221** | **0.1819** |

---

### MI3DOR
| Method   | NN ↑  | FT ↑  | ST ↑  | F ↑   | DCG ↑  | ANMRR ↓ |
|----------|-------|-------|-------|-------|--------|---------|
| Ours-SF  | 0.7710 | 0.6630 | 0.7960 | 0.1560 | 0.6920 | 0.3140 |
| Ours-SA  | **0.7970** | **0.6700** | **0.8130** | **0.1560** | **0.7000** | **0.3110** |



## 🛠️ Installation

## Prerequisites
- python == 3.8.5

Please make sure you have the following libraries installed:
- numpy
- torch>=1.4.0
- torchvision>=0.5.0



