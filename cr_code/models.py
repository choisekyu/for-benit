import pickle
from pathlib import Path

import lightgbm as lgb
import optuna
import torch
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import mean_squared_error


class AutoEncoder(torch.nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.BatchNorm1d(4),
            torch.nn.Linear(4, 1),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(1, 4),
            torch.nn.BatchNorm1d(4),
            torch.nn.Linear(4, 4)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class AutoEncoder1(torch.nn.Module):

    def __init__(self, input_size):
        super(AutoEncoder1, self).__init__()
        self.encoder1 = torch.nn.Sequential(
            torch.nn.Linear(input_size-2, 16384),
            #torch.nn.BatchNorm1d(16384),
            torch.nn.ReLU(),
            torch.nn.Linear(16384, 2048),
            #torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 512),
            #torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
        )
        self.encoder2 = torch.nn.Sequential(
            torch.nn.Linear(514, 32)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(32, 512),
            #torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2048),
            #torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 16384),
            #torch.nn.BatchNorm1d(16384),
            torch.nn.ReLU(),
            torch.nn.Linear(16384, input_size),
        )

    def forward(self, x):
        x1 = x[:, :-2]
        x2 = x[:, -2:]
        x = self.encoder1(x1)
        x = self.encoder2(torch.concat([x, x2], dim=1))
        x = self.decoder(x)
        return x


def _postprocess(model, x_test, y_test, prefix: str, save_path: Path | str):
    if type(model).__name__ == 'method':
        y_pred = model(x_test)
    else:
        y_pred = model.predict(x_test)

    if save_path:
        if isinstance(save_path, str):
            save_path = Path(save_path)

        save_path = (save_path / f'{prefix}_{save_path.name}').as_posix()
        with open(f'{save_path}.pkl', 'wb') as pkl:
            pickle.dump(model, pkl)

        y_pred = y_pred, save_path

    return y_pred


def ridge(x_train, y_train, x_test,
          y_test=None,
          type: str = 'reg',
          mb: bool = False,
          save_path: Path | str = None):
    if type == 'reg':
        model = Ridge(alpha=1.0)
    elif type == 'classifier':
        model = RidgeClassifier(alpha=1.0)
    else:
        print('reg or classifier')
        return

    model.fit(x_train, y_train)
    prefix = f'ridge_{type}_mb' if mb else f'ridge_{type}_trend'
    return _postprocess(model, x_test, y_test, prefix, save_path)


def lgbm(x_train, y_train, x_test,
         y_test=None,
         x_list='auto',
         device: str = 'cpu',
         objective: str = 'regression',
         type: str = 'reg',
         mb: bool = False,
         save_path: Path | str = None):
    y_train = y_train.flatten()
    def lgbm_objective(trial):
        # LightGBM 파라미터 범위 설정
        params = {
            'objective': objective,  # regression or binary or etc...
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': trial.suggest_categorical(
                'boosting_type', ['gbdt', 'dart']),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'learning_rate': trial.suggest_float(
                'learning_rate', 0.001, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
            'feature_fraction': trial.suggest_float('feature_fraction',0.4,1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction',0.4,1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
        # LightGBM 학습
        if type == 'reg':
            model = lgb.LGBMRegressor
        elif type == 'classifier':
            model = lgb.LGBMClassifier

        model = model(**params, max_bin=15, device=device, gpu_use_dp=False)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_train)
        rmse = mean_squared_error(y_train, y_pred, squared=False)
        return rmse

    # Optuna 스터디 생성 및 최적화 수행
    study = optuna.create_study(direction='minimize')
    study.optimize(lgbm_objective, n_trials=30)

    # 결과 출력
    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('  Value: {:.3f}'.format(trial.value))
    print('  Params: ')
    for k, value in trial.params.items():
        print('    {}: {}'.format(k, value))

    num_round = 100
    params = trial.params
    lgb_train = lgb.Dataset(x_train, y_train, feature_name=x_list)
    model = lgb.train(params, lgb_train, num_round)
    prefix = f'lgbm_{type}_mb' if mb else f'lgbm_{type}_trend'
    return _postprocess(model, x_test, y_test, prefix, save_path)
