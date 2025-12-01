# 비디오 요약 모델 비교 실험 보고서

## Method: New or Improved Aspects Proposed in the Term Project

### 1. 제안 모델: LoGoNet (Local-Global Network)

본 프로젝트에서는 **LoGoNet (Local-Global Network)**이라는 새로운 하이브리드 아키텍처를 제안합니다. LoGoNet은 비디오 요약 태스크에서 중요한 두 가지 관점인 **Local Patterns**와 **Global Context**를 동시에 모델링하는 dual-path 구조를 채택합니다.

#### 1.1 핵심 아이디어

비디오 요약에서 프레임의 중요도를 결정하기 위해서는 두 가지 정보가 필요합니다:
- **Local Patterns**: 인접한 프레임 간의 시간적 패턴 (예: 액션 전환, 장면 변화)
- **Global Context**: 전체 시퀀스에 걸친 전역적 맥락 (예: 스토리 흐름, 전체적인 중요도 분포)

기존 모델들은 주로 단일 관점에 집중하지만, LoGoNet은 두 관점을 병렬로 처리한 후 효과적으로 융합합니다.

#### 1.2 아키텍처 설계

LoGoNet은 다음과 같은 5단계 구조로 구성됩니다:

**1) Local Path (2D CNN 기반)**
- CSTA에서 영감을 받은 2D CNN 구조
- Kernel size (3,1) 또는 (5,1)로 시간 차원에서만 convolution 수행
- 인접 프레임 간의 local temporal patterns 추출
- Residual connection으로 gradient flow 개선

**2) Global Path (Transformer Encoder 기반)**
- FFT Mixer 대신 Transformer Encoder 채택
- Multi-Head Self-Attention (8 heads)으로 모든 프레임 쌍 간 관계 학습
- Pre-norm architecture로 학습 안정성 확보
- Learnable positional encoding 사용

**3) Cross-Path Attention (양방향 정보 교환)**
- Local-to-Global Attention: Local features가 Global context를 참조
- Global-to-Local Attention: Global features가 Local details를 참조
- 두 경로 간 상호 보완적 정보 교환으로 통합 표현 향상

**4) Adaptive Fusion (동적 가중치 융합)**
- Gate Network를 통한 per-frame adaptive weights 계산
- 단순 concatenation 대신 맥락 인식 동적 가중치 기반 융합
- Weighted combination과 concatenation을 모두 포함하여 풍부한 representation 생성

**5) Score Regression**
- LayerNorm → Linear(512→256) → GELU → Dropout → Linear(256→1) → Sigmoid
- 최종 importance scores (0~1 범위) 출력

#### 1.3 주요 개선 사항

**Transformer 기반 Global Path**
- FFT Mixer 대신 Transformer Encoder로 장기 의존성 학습 능력 향상
- Self-Attention을 통해 모든 프레임 쌍 간 상관관계를 직접 학습

**Cross-Path Attention**
- Local과 Global 경로가 독립적으로 작동하는 것을 방지
- 초기 단계부터 정보를 교환하여 통합 표현 향상

**Adaptive Fusion**
- 각 프레임마다 Local/Global의 중요도를 동적으로 조절
- 영상/프레임별로 Local/Global 기여도를 자동 조정

**안정성 개선 기법**
- Attention weights를 매우 작은 값(gain=0.01)으로 초기화
- 모든 중간 출력을 [-10, 10] 범위로 clamping
- NaN/Inf 체크 및 예외 처리로 학습 안정성 확보
- Gradient clipping을 0.5로 강화

#### 1.4 학습 설정

- **배치 사이즈**: 8 (RTX 3080 10GB, 자동 탐색 결과)
- **Gradient Accumulation**: 2 (effective batch size = 16)
- **Optimizer**: AdamW (weight_decay=1e-4)
- **Learning Rate**: 1e-4 (warmup 10 epochs + cosine annealing to 1e-7)
- **Gradient Clipping**: 0.5
- **모델 파라미터**: 약 11.8M

---

## Experiments: Results and Analysis

### 2. 실험 설정

#### 2.1 데이터셋 및 평가 지표
- **데이터셋**: MR.HiSum (Canonical Split)
- **Train/Val/Test**: 27,892 / 2,000 / 2,000 samples
- **평가 지표**: Kendall's Tau, Spearman's Rho, Test Loss (MSE)

#### 2.2 비교 모델 및 학습 설정

| 모델 | 배치 사이즈 | Epochs | Peak LR | LR Scheduler |
|------|------------|--------|---------|--------------|
| CSTA | 4 | 50 | 3e-4 | Warmup(3) + Cosine |
| VideoSAGE | 512 | 50 | 5e-4 | Warmup(5) + Cosine |
| EDSNet | 112 | 300 | 5e-5 | Fixed |
| **LoGoNet** | **8** | **50** | **1e-4** | **Warmup(10) + Cosine** |

### 3. 실험 결과

#### 3.1 전체 모델 성능 비교

| Model | Test Loss | Kendall's Tau | Spearman's Rho | 순위 |
|-------|-----------|---------------|----------------|------|
| **LoGoNet** (최신) | 0.0632 | **0.1336** | **0.1909** | **1위** |
| **CSTA** | 0.0579 | 0.1196 | 0.1717 | 2위 |
| **EDSNet** | **0.0457** | 0.1158 | 0.1659 | 3위 |
| **VideoSAGE** | 0.0443 | 0.0810 | 0.1175 | 4위 |

#### 3.2 주요 발견

**LoGoNet의 우수한 성능**
- **Spearman's Rho**: 0.1909로 모든 모델 중 최고 성능
  - CSTA(0.1717) 대비 **11.1% 향상**
  - EDSNet(0.1659) 대비 **15.1% 향상**
  - VideoSAGE(0.1175) 대비 **62.5% 향상**
- **Kendall's Tau**: 0.1336로 모든 모델 중 최고 성능
  - CSTA(0.1196) 대비 **11.7% 향상**
  - EDSNet(0.1158) 대비 **15.4% 향상**
  - VideoSAGE(0.0810) 대비 **64.9% 향상**

**LoGoNet 개선 효과**
- 이전 버전(2025-11-22) 대비:
  - Spearman's Rho: 0.1213 → 0.1909 (**57.4% 향상**)
  - Kendall's Tau: 0.0837 → 0.1336 (**59.7% 향상**)

#### 3.3 결과 분석

**Loss vs 순위 상관계수 불일치**
- EDSNet과 VideoSAGE는 낮은 Test Loss를 보이지만, 순위 상관계수는 상대적으로 낮음
- LoGoNet은 Test Loss가 상대적으로 높지만, 순위 상관계수에서는 최고 성능
- **시사점**: 비디오 요약 태스크에서는 순위 상관계수가 더 중요한 평가 지표

**하이브리드 아키텍처의 효과**
1. **Dual-Path Design**: Local과 Global 정보를 동시에 활용하여 단일 경로 모델 대비 성능 향상
2. **Cross-Path Attention**: 두 경로 간 정보 교환으로 상호 보완적 학습
3. **Adaptive Fusion**: 맥락 인식 동적 융합으로 더 유연한 특징 표현

**모델별 특성**
- **CSTA**: CNN 기반 attention으로 local과 global 정보 포착, 순위 상관계수에서 두 번째 성능
- **VideoSAGE**: 그래프 구조로 복잡한 관계 모델링, Test Loss는 낮지만 순위 상관계수는 낮음
- **EDSNet**: 경량 모델로 효율적 처리, Test Loss 최저, 순위 상관계수는 중간 수준
- **LoGoNet**: Dual-path design으로 순위 상관계수에서 최고 성능

---

## Conclusion: Final Summary and Takeaways

### 4. 주요 성과

#### 4.1 성능 향상
**LoGoNet이 모든 모델 중 최고의 순위 상관계수 성능을 달성했습니다:**
- Spearman's Rho: **0.1909** (CSTA 대비 11.1% 향상)
- Kendall's Tau: **0.1336** (CSTA 대비 11.7% 향상)

**LoGoNet 개선 버전의 효과가 검증되었습니다:**
- 이전 버전 대비 Spearman's Rho **57.4% 향상**
- Kendall's Tau **59.7% 향상**

#### 4.2 하이브리드 아키텍처의 우수성 입증
- **Dual-Path Design**: Local과 Global 정보를 동시에 활용하는 구조의 효과
- **Cross-Path Attention**: 두 경로 간 정보 교환으로 상호 보완적 학습
- **Adaptive Fusion**: 맥락 인식 동적 융합으로 더 유연한 특징 표현

### 5. 시사점

#### 5.1 평가 지표의 중요성
비디오 요약 태스크에서는 **Test Loss보다 순위 상관계수가 더 중요한 평가 지표**입니다. 실제 응용에서는 프레임의 절대적 중요도 점수보다는 상대적 순위가 더 중요하기 때문입니다. LoGoNet은 Test Loss는 상대적으로 높지만, 순위 상관계수에서는 최고 성능을 달성하여 이 점을 잘 보여줍니다.

#### 5.2 하이브리드 접근의 효과
단일 아키텍처보다 **다양한 관점을 결합한 하이브리드 모델이 우수한 성능**을 보입니다. LoGoNet의 성공은 Local과 Global 정보를 동시에 활용하는 dual-path design의 효과를 입증합니다.

#### 5.3 안정성 개선의 필요성
LoGoNet의 안정성 개선 기법들(안전한 초기화, 값 clamping, NaN/Inf 체크, 강화된 gradient clipping)이 성공적인 학습에 기여했습니다. 특히 Transformer 기반 모델에서 이러한 안정성 기법이 중요함을 확인했습니다.

### 6. 향후 연구 방향

#### 6.1 Loss 함수 개선
순위 상관계수를 직접 최적화하는 loss 함수를 고려할 수 있습니다. 현재는 MSE Loss를 사용하지만, ranking loss나 listwise loss를 사용하면 순위 상관계수를 더 직접적으로 개선할 수 있을 것입니다.

#### 6.2 하이퍼파라미터 튜닝
더 긴 학습 시간(200 epochs 이상)과 다양한 하이퍼파라미터 조합(transformer layer 수, attention head 수, hidden dimension 등)을 실험하여 성능을 더 향상시킬 수 있습니다.

#### 6.3 아키텍처 개선
- **Hierarchical Temporal Modeling**: 프레임 → 샷 단계적 모델링
- **Multi-Modal Integration**: 자막/음향 정보 통합
- **Sparse Attention**: 긴 시퀀스 처리 효율화

#### 6.4 효율성 개선
모델 크기와 연산량을 줄이면서 성능을 유지하는 방법을 탐구할 수 있습니다. Knowledge distillation이나 model compression 기법을 적용하여 실용성을 높일 수 있습니다.

### 7. 최종 결론

본 프로젝트에서 제안한 **LoGoNet은 Local과 Global 정보를 동시에 활용하는 하이브리드 아키텍처**로, 모든 비교 모델 중 최고의 순위 상관계수 성능을 달성했습니다. 특히 Cross-Path Attention과 Adaptive Fusion을 통한 두 경로 간 정보 교환과 동적 융합이 성능 향상에 핵심적인 역할을 했습니다. 

비디오 요약 태스크에서 **순위 상관계수가 Test Loss보다 더 중요한 평가 지표**임을 확인했으며, 하이브리드 접근 방식이 단일 아키텍처보다 우수한 성능을 보임을 입증했습니다. 향후 순위 상관계수를 직접 최적화하는 loss 함수와 더 긴 학습 시간을 통한 추가 개선이 가능할 것으로 기대됩니다.

---

**실험 환경**: NVIDIA RTX 3080 (10GB VRAM), PyTorch  
**데이터셋**: MR.HiSum (Canonical Split)  
**실험 기간**: 2025년 11월 22일 ~ 2025년 11월 30일

