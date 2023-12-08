from EDGE import EDGE
from torch.utils.data import DataLoader

model = EDGE("baseline", "./weights/train-1.pt")
model.eval()

train_data_loader = DataLoader("./data/test/gBR_sBM_cAll_d04_mBR0_ch02_slice0.npy")

train_data_loader = model.accelerator.prepare(train_data_loader)

# directory for optionally saving the dances for eval
data_tuple = None, all_cond[i], all_filenames[i]

model.render_sample(
    data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, render=not opt.no_render
)
