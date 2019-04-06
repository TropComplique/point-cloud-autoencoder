import torch
import json
from torch.utils.data import DataLoader
from input_pipeline import PointClouds
from trainer import Trainer


NUM_EPOCHS = 24
BATCH_SIZE = 32
PATH = 'models/run00.pth'
DEVICE = torch.device('cuda:0')
TRAIN_PATH = ''
VAL_PATH = ''
labels = ['02691156']
dataset_path = '/home/dan/datasets/shape_net_core_uniform_samples_2048/'
TRAIN_LOGS = 'models/run00.json'


def train_and_evaluate():

    train = PointClouds(dataset_path, labels, is_training=True)
    val = PointClouds(dataset_path, labels, is_training=False)

    train_loader = DataLoader(
        dataset=train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True
    )

    num_steps = NUM_EPOCHS * (len(train) // BATCH_SIZE)
    model = Trainer(num_steps, DEVICE)
    # model.network.to(DEVICE)

    i = 0
    logs = []
    text = 'e: {0}, i: {1}, loss: {2:.3f}'

    for e in range(NUM_EPOCHS):

        model.network.train()
        for x in train_loader:

            x = x.to(DEVICE)
            loss = model.train_step(x)

            i += 1
            log = text.format(e, i, loss)
            print(log)
            logs.append(loss)

        eval_losses = []
        model.network.eval()
        for batch in val_loader:

            x = x.to(DEVICE)
            loss = model.evaluate(x)
            eval_losses.append(loss)

        eval_losses = {k: sum(d[k] for d in eval_losses)/len(eval_losses) for k in losses.keys()}
        eval_losses.update({'type': 'eval'})
        print(eval_losses)
        logs.append(eval_losses)

        model.save(PATH)
        with open(TRAIN_LOGS, 'w') as f:
            json.dump(logs, f)


train_and_evaluate()
