
import torch
import numpy as np

def test_epoch(model, loader ,criterion,args):
    model.eval()
    losses, outputs, targets = [], [] , []
    for batch in loader:
        batch = batch.to(args.device)
        out = model(batch)
        outputs.append(out)
        targets.append(batch.y)
        loss = criterion(out,batch.y)
        losses.append(loss.item())

    return np.array(losses).mean(),torch.cat(outputs), torch.cat(targets)