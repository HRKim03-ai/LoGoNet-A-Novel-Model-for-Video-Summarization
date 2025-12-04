# Video Summarization Models Comparison

비디오 요약 태스크를 위한 4가지 딥러닝 모델 구현 및 비교 실험 프로젝트

## 📋 프로젝트 개요

본 프로젝트는 MR.HiSum 데이터셋을 사용하여 4가지 비디오 요약 모델(CSTA, VideoSAGE, EDSNet, LoGoNet)을 구현하고 성능을 비교합니다. 특히 **LoGoNet**이라는 새로운 하이브리드 아키텍처를 제안하여 최고 성능을 달성했습니다.

## 🎯 주요 성과

### 실험 결과 (Test Set)

| Model | Test Loss | Kendall's Tau | Spearman's Rho | 순위 |
|-------|-----------|---------------|----------------|------|
| **LoGoNet** (제안) | 0.0632 | **0.1336** | **0.1909** | **1위** |
| **CSTA** | 0.0579 | 0.1196 | 0.1717 | 2위 |
| **EDSNet** | **0.0457** | 0.1158 | 0.1659 | 3위 |
| **VideoSAGE** | 0.0443 | 0.0810 | 0.1175 | 4위 |

**주요 발견:**
- **LoGoNet**이 모든 모델 중 최고의 순위 상관계수 성능 달성
  - Spearman's Rho: 0.1909 (CSTA 대비 11.1% 향상)
  - Kendall's Tau: 0.1336 (CSTA 대비 11.7% 향상)
- 비디오 요약 태스크에서는 순위 상관계수가 Test Loss보다 더 중요한 평가 지표임을 확인

## 🏗️ 모델 아키텍처

### 1. CSTA (CNN-based Spatiotemporal Attention)
- **논문**: CVPR 2024
- **구조**: 2D CNN + Attention 기반
- **특징**: CNN attention으로 local과 global 정보 포착

### 2. VideoSAGE (Video Summarization with Graph Representation Learning)
- **논문**: CVPRW 2024
- **구조**: GCN 기반 그래프 모델링
- **특징**: 프레임을 그래프 노드로 모델링하여 복잡한 관계 학습

### 3. EDSNet (Efficient-DSNet)
- **논문**: arXiv 2024
- **구조**: Token Mixer (MLP-Mixer) 기반 경량 모델
- **특징**: Attention 없이도 효율적으로 전역 정보 처리

### 4. LoGoNet (Local-Global Network) - **제안 모델**
- **핵심 아이디어**: Local Patterns와 Global Context를 동시에 모델링하는 하이브리드 아키텍처
- **구조**:
  1. **Local Path**: 2D CNN으로 인접 프레임 간 시간적 패턴 추출
  2. **Global Path**: Transformer Encoder로 전체 시퀀스 맥락 학습
  3. **Cross-Path Attention**: 두 경로 간 양방향 정보 교환
  4. **Adaptive Fusion**: 동적 가중치 기반 특징 융합
  5. **Score Regression**: 최종 importance scores 예측
- **모델 파라미터**: 약 11.8M
- **주요 개선사항**:
  - Transformer 기반 Global Path로 장기 의존성 학습 향상
  - Cross-Path Attention으로 상호 보완적 학습
  - Adaptive Fusion으로 맥락 인식 동적 융합

## 📁 프로젝트 구조

```
.
├── README.md                 # 프로젝트 설명서
├── TERM_PROJECT_REPORT.md    # 실험 보고서
├── train.py                  # 통합 학습 스크립트
├── test.py                   # 테스트 스크립트
├── dataset.py                # 데이터셋 로더
├── models/                   # 모델 구현
│   ├── csta.py
│   ├── videosage.py
│   ├── edsnet.py
│   └── logonet.py           # 제안 모델
├── data/                     # 데이터셋 관련 유틸리티
├── evaluation/               # 평가 지표
├── utils/                    # 유틸리티 함수
├── results/                  # 실험 결과
└── (checkpoints/)            # 로컬에서만 사용되는 모델 체크포인트 (제출/깃허브에는 포함하지 않음)
```

## 🚀 설치 및 실행

### 환경 설정

```bash
# Conda 환경 활성화
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate mrhisum

# 필요한 패키지 설치
pip install torch torchvision scipy tqdm h5py
```

### 데이터셋 준비

데이터셋은 `dataset/` 디렉토리에 다음 파일들이 필요합니다:
- `mr_hisum.h5`: HDF5 형식의 비디오 특징 및 GT
- `mr_hisum_split.json`: Train/Val/Test Split 정보

### 모델 학습

```bash
# LoGoNet 학습 (제안 모델)
python train.py \
    --model logonet \
    --batch_size 8 \
    --epochs 50 \
    --lr 1e-4 \
    --max_grad_norm 0.5

# 다른 모델 학습
python train.py --model csta --batch_size 4 --epochs 50
python train.py --model videosage --batch_size 256 --epochs 50
python train.py --model edsnet --batch_size 112 --epochs 300
```

### 모델 테스트

```bash
# 학습된 모델 테스트
python test.py \
    --model logonet \
    --checkpoint /path/to/logonet_best.pth \  # 로컬에 저장된 체크포인트 경로
    --batch_size 8 \
    --output_file results/logonet_test.txt
```

## 📊 실험 설정

### 하이퍼파라미터

| 모델 | 배치 사이즈 | Epochs | Peak LR | LR Scheduler |
|------|-----------|--------|---------|--------------|
| CSTA | 4 | 50 | 3e-4 | Warmup(3) + Cosine annealing|
| VideoSAGE | 512 | 50 | 5e-4 | Warmup(5) + Cosine annealing |
| EDSNet | 112 | 300 | 5e-5 | Fixed |
| LoGoNet | 8 | 50 | 1e-4 | Warmup(10) + Cosine annealing |

### 평가 지표

- **Kendall's Tau**: 순위 상관계수 (두 순위 간 일치도 측정)
- **Spearman's Rho**: 순위 상관계수 (모노톤 관계 측정)
- **Test Loss**: MSE Loss

## 📝 주요 파일 설명

- `TERM_PROJECT_REPORT.md`: 상세한 실험 보고서 및 결과 분석
- `train.py`: 통합 학습 스크립트 (모든 모델 지원)
- `test.py`: 테스트 셋 평가 스크립트
- `models/logonet.py`: 제안 모델 LoGoNet 구현
- `results/`: 실험 결과 파일들

## 🔧 기술 스택

- **Framework**: PyTorch
- **Hardware**: NVIDIA RTX 3080 (10GB VRAM)
- **Dataset**: MR.HiSum (Canonical Split)
- **Mixed Precision**: torch.cuda.amp (VRAM 효율)

## 📚 참고 문헌

1. MR.HiSum Dataset
2. Rethinking the Evaluation of Video Summaries(CVPR 2019)
3. CSTA: CNN-based Spatiotemporal Attention for Video Summarization (CVPR 2024)
4. VideoSAGE: Video Summarization with Graph Representation Learning (CVPR 2024)
5. EDSNet: Efficient-DSNet (arXiv 2024)

## 📄 라이선스

본 프로젝트는 학술 연구 목적으로 제작되었습니다.

## 👤 작성자

2025년 11월 22일 ~ 2025년 11월 30일

---

**참고**: WandB(Weights & Biases)는 선택사항이며, `--use_wandb` 플래그로 활성화할 수 있습니다. 기본적으로는 사용하지 않습니다.

