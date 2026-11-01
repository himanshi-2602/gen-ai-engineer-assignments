| Model | Parameters | Test accuracy | Test loss | Best val accuracy | Best epoch | Train time |
|---|---|---|---|---|---|---|
| baseline | 1,148,874 | **91.77%** | 0.2448 | 92.38% | 29/30 | 10.4 min |
| residual | 1,190,922 | **91.79%** | 0.2387 | 92.58% | 30/30 | 14.1 min |

| Class | baseline | residual |
|---|---|---|
| airplane | 92.90% | 93.40% |
| automobile | 96.30% | 95.60% |
| bird | 86.70% | 85.50% |
| cat | 81.90% | 81.10% |
| deer | 92.80% | 93.60% |
| dog | 87.20% | 88.10% |
| frog | 95.40% | 94.60% |
| horse | 93.70% | 94.30% |
| ship | 96.10% | 96.60% |
| truck | 94.70% | 95.10% |

**residual vs baseline: +0.02 percentage points of test accuracy.**
