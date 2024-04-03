import argparse
import pickle
import shutil

import torch
from sklearn.preprocessing import StandardScaler

from db import CRDB
from models import AutoEncoder, AutoEncoder1, ridge
from preprocess import query_to_df, rearrange_by_time, make_dataset


def _train(x_train, x_test, save_path: Path | str = None, mb: bool = False
) -> torch.tensor:
    if mb:
        type = 'mb'
        model = AutoEncoder().cuda()
    else:
        type = 'trend'
        model = AutoEncoder1(x_train.shape[-1]).cuda()

    if save_path:
        if isinstance(save_path, str):
            save_path = Path(save_path)

        if save_path.suffix == '':
            save_path = save_path.parent / f'{save_path.stem}.pt'

        best_path = save_path.parent / f'best_ae_{type}_{save_path.name}'
        last_path = save_path.parent / f'last_ae_{type}_{save_path.name}'

    model.train(True)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_min = 1e+5
    for epoch in range(20_000):
        optimizer.zero_grad()
        result = model(x_train)
        #loss = torch.sqrt(loss_fn(result, x_train) + 1e-7)
        loss = torch.sqrt(loss_fn(result, x_train))
        loss.backward()
        optimizer.step()
        loss = loss.item()
        if epoch % 100 == 0:
            print(epoch+1, loss)

        if save_path and loss_min > loss:
            loss_min = loss
            if loss_min < 1:
                torch.save(model, best_path)

    if save_path:
        torch.save(model, last_path)

    model.eval()
    with torch.no_grad():
        result = model(x_test)

    return result.cpu()


def train_ae(data):
    for rm, rm_value in data.items():
        for key in ['motor', 'wire']:
            dataset = make_dataset(rm_value[key])
            for k, x_train in dataset.items():
                x_train = torch.from_numpy(x_train).float().cuda()
                name = f'{rm}_{key}_{k}'
                result = _train(x_train, x_train, f'results/{name}', mb=True)
                result = torch.sqrt(torch.sum((result-x_train)**2, 1))


def train_ml(data):
    for rm, rm_value in data.items():
        for key in ['motor', 'wire']:
            dataset = make_dataset(rm_value[key])
            for k, x_train in dataset.items():
                name = f'{rm}_{key}_{k}'
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                # need y_train: label
                y_pred, filename = ridge(x_train, y_train, x_train,
                                         mb=True, save_path=f'results/{name}')
                with open(f'{filename}_scaler.pkl', 'wb') as pkl:
                    pickle.dump(scaler, pkl)


def predict(model_path: str, data) -> torch.tensor:
    data = torch.from_numpy(data).float().cuda()
    model = torch.load(model_path).cuda()
    model.eval()
    with torch.no_grad():
        result = model(data)
        result = torch.sqrt(torch.sum((result-data)**2, 1)).detach().cpu()

    return result


if __name__ == '__main__':
    # load data
    crdb = CRDB('server', 'database', 'username', 'password', 'table')
    data = crdb.query('2024-01-01', '2024-02-01')
    data = query_to_df(data, crdb.columns)
    data = {k: data[data['EQUIPMENT_ID'].str.contains(k)].reset_index(drop=True)
            for k in ['RM01', 'RM05']}
    data = {k: rearrange_by_time(v) for k, v in data.items()}

    # train
    train_ae(data)
