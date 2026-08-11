# lnn1step

## LNN-SCA: Liquid Neural Networks for Side-Channel Security Assessment

Implementation of CfC-based Liquid Neural Networks for deep learning side-channel analysis (DL-SCA), evaluated on the ASCAD benchmark dataset.

## Files

- `lnn_sca.py` — LNN-SCA model definition (FC compression + CfC cell + classifier)
- `rank_test_lnn.py` — Key-rank evaluation script (ASCAD `full_ranks()` protocol)
- `lnn_baseline.pth` — Pretrained model checkpoint (single-step CfC, clean traces)

