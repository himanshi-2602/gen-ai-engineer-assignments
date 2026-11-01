| Run | Parameters | Test accuracy | Test loss | Best val accuracy | Best epoch | Train time |
|---|---|---|---|---|---|---|
| residual | 1,190,922 | **91.79%** | 0.2387 | 92.58% | 30/30 | 14.1 min |
| residual_amp | 1,190,922 | **92.07%** | 0.2397 | 92.80% | 30/30 | 13.1 min |

| Class | residual | residual_amp |
|---|---|---|
| airplane | 93.40% | 92.50% |
| automobile | 95.60% | 96.50% |
| bird | 85.50% | 86.90% |
| cat | 81.10% | 81.30% |
| deer | 93.60% | 94.50% |
| dog | 88.10% | 88.40% |
| frog | 94.60% | 95.00% |
| horse | 94.30% | 94.30% |
| ship | 96.60% | 96.40% |
| truck | 95.10% | 94.90% |

**residual_amp vs residual: +0.28 percentage points of test accuracy.**
