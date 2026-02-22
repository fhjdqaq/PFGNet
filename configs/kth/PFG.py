method = 'PFG'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type='None'
hid_S = 64
N_T = 6
N_S = 2
# training
lr = 2e-4
drop_path = 0.1
batch_size = 2  # bs = 4 x 2GPUs
sched = 'onecycle'