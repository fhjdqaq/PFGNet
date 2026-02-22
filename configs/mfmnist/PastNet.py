# configs/mfmnist/PastNet.py
method = 'PastNet'

# I/O（Fashion-MNIST 默认：10 -> 10，分辨率 64x64，单通道）
pre_seq_length = 10
aft_seq_length = 10
in_shape = (pre_seq_length, 1, 64, 64)  # (T, C, H, W)

# PastNet core
hid_T = 256
N_T = 8
res_units = 32
res_layers = 4
K = 512
D = 64

# Fourier / FPG
fourier_num_blocks   = 8
fourier_softshrink   = 0.0
fourier_use_bias     = False
fourier_double_skip  = False
fourier_checkpoint   = False

# Train
batch_size = 16
val_batch_size = 4
epoch = 100
lr = 1e-3
optim = 'adamw'
sched = 'onecycle'
num_workers = 4
