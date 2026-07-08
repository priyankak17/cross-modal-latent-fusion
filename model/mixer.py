import torch
import os

mix_1 = torch.load('./chunked_pts/celeba_latents_final.pt').to('cuda')
mix_2 = torch.load('./sketch_chunk_pts/celeba_latents_final.pt').to('cuda')
combo = torch.tensor([]).to('cuda')
for i in range(mix_1.shape[0]):
    combo = torch.cat([combo, torch.cat([(mix_2[i]).unsqueeze(0)[:, :5,:], (mix_2[i]).unsqueeze(0)[:, 5:, :]], dim=1)])
torch.save(combo, 'combo.pt')