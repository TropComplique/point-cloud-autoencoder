import torch
import json
from torch.utils.data import DataLoader
from input_pipeline import PointClouds
from trainer import Trainer


NUM_EPOCHS = 24
BATCH_SIZE = 50
PATH = 'models/run00.pth'
DEVICE = torch.device('cuda:0')
TRAIN_PATH = ''
VAL_PATH = ''
TRAIN_LOGS = 'models/run00.json'


def train_and_evaluate():

    train = PointClouds()
    val = PointClouds(is_training=False)

    train_loader = DataLoader(
        dataset=train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True
    )

    num_steps = NUM_EPOCHS * (len(train) // BATCH_SIZE)
    model = Trainer(num_steps)
    model.network.to(DEVICE)

    i = 0
    logs = []
    text = 'e: {0}, i: {1}, loss: {2:.3f}'

    for e in range(NUM_EPOCHS):

        model.network.train()
        for x in train_loader:

            x = x.to(DEVICE)
            loss = model.train_step(x)

            i += 1
            log = text.format(e, i, loss.item())
            print(log)
            logs.append({n: float(v.item()) for n, v in losses.items()})

        eval_losses = []
        model.network.eval()
        for batch in val_loader:

            x = x.to(DEVICE)
            loss = model.train_step(x)
            eval_losses.append({n: float(v.item()) for n, v in losses.items()})

        eval_losses = {k: sum(d[k] for d in eval_losses)/len(eval_losses) for k in losses.keys()}
        eval_losses.update({'type': 'eval'})
        print(eval_losses)
        logs.append(eval_losses)

        model.save(PATH)
        with open(TRAIN_LOGS, 'w') as f:
            json.dump(logs, f)


train_and_evaluate()
