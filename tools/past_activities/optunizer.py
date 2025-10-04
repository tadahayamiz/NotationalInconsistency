import numpy as np
import pandas as pd
import optuna

class Optunizer:
    def __init__(self, direction, data=None, target=None, cross_val=True, k_val=5,
         val_size=0.2, n_trials=100, sampler=None, pruner=None, cross_test=False,
        k_test=5, test_size=0.2):
        """
        Parameters
        ----------
        data: data type or None
            Datas used to tasks in this class.
        target: target type or None
            Targets used to tasks in this class. 
            Targets can be included in data as tuple.
        """
        self.direction = direction
        self.data = data
        self.target = target
        self.cross_val = cross_val
        self.k_val = k_val
        self.val_size = val_size
        self.n_trials = n_trials
        self.sampler = sampler
        self.pruner = pruner
        self.cross_test = cross_test
        self.k_test = k_test
        self.test_size = test_size
        self.is_training = True
        self.study = None
        self.trial = None


    def get_optuna_param(self, param_name, suggest_type, param_range):
        """
        Get hyper parameter, according to whether training or evaluation.

        Parameters
        ----------
        param_name: str
            Name of parameter which is set to trial
        suggest_type: str
            Function to use to suggest parameter
            'uniform': suggest_uniform()
            'int': suggest_int() etc.
        param_range: array_like
            Parameter range. Input of suggest_~
        """
        if self.is_training:
            if suggest_type == 'categorical':
                return self.trial.suggest_categorical(param_name, param_range)
            elif suggest_type == 'discrete_uniform':
                return self.trial.suggest_discrete_uniform(param_name, *param_range)
            elif suggest_type == 'int':
                return self.trial.suggest_int(param_name, *param_range)
            elif suggest_type == 'float':
                return self.trial.suggest_float(param_name, *param_range)
            elif suggest_type == 'loguniform':
                return self.trial.suggest_loguniform(param_name, *param_range)
            elif suggest_type == 'uniform':
                return self.trial.suggest_uniform(param_name, *param_range)
            else:
                raise ValueError(f"Unsupported type of suggestion: {suggest_type}")
        else:
            return self.study.best_params[param_name]

    def get_hparams(self):
        """
        Arbitrary function to get hyper parameter of model.

        Returns
        -------
        hparam: any type
            Hyper parameter.
        """
        raise NotImplementedError

    def adjust_params(self, hparams, data_train=None):
        """
        Adjust parameters for training data.

        Parameters
        ----------
        hparams: any type
            Hyper parameter used when adjusting parameters.

        data_train: Data type/ array_like of int or None
            Data, or indices of self.data (depending on self._adjust_params)
            used when adjusting parameter.
            If None, indices of whole self.data is substituted.
        """
        if data_train is None:
            data_train = np.arange(len(self.data))
        return self._adjust_params(hparams, data_train)

    def _adjust_params(self, hparams, data_train):
        raise NotImplementedError

    def train_test_split(self, test_size, data=None):
        """
        Splits data into train set and test set.

        Parameters
        ----------
        test_size: float
            Fraction of test set in whole set.
        data; data type or array_like of int.
            Data or indices of data to split.
            If None, indices of whole self.data is substituted.
        
        Returns (data_train, data_test)
        -------
        data_train: data type or array_like of int.
            Data or indices of train data.
        data_test: data type or array_like of int.
            Data or indices of test data.
        """
        if data is None:
            data = np.arange(len(self.data))
        return self._train_test_split(test_size, data)

    def _train_test_split(self, test_size, data):
        raise NotImplementedError
        

    def k_fold_split(self, k, data=None):
        """
        Yields k-folded train/test data.

        Parameters
        ----------
        k: int
            Number of splits.
        data: data type or array_like of int
            Data or indices of data to k-fold
            If None, indices of whole self.data is substituted.

        Yields (data_train, data_test)
        ------
        data_train: data type or array_like of int.
            Data or indices of folded train data.
        data_test: data type or array_like of int.
            Data or indices of folded test data.
        """
        if data is None:
            data = np.arange(len(self.data))
        return self._k_fold_split(k, data)

    def _k_fold_split(self, k, data):
        raise NotImplementedError

    def eval_param(self, is_objective, data_test=None):
        if data_test is None:
            data_test = np.arange(len(self.data))
        return self._eval_param(is_objective, data_test)

    def _eval_param(self, is_objective, data_test):
        raise NotImplementedError

    def adjust_hparams(self, data_trainval=None):
        self.is_training = True
        def objective(trial):
            self.trial = trial
            hparams = self.get_hparams()
            
            if self.cross_val:
                fold = self.k_fold_split(self.k_val, data=data_trainval)
            else:
                fold = [self.train_test_split(test_size=self.val_size,
                    data=data_trainval)]
            results = 0
            for data_train, data_val in fold:
                self.adjust_params(hparams, data_train)
                result = self.eval_param(is_objective=True, data_test=data_val)
                results += result
            return result
        
        self.study = optuna.create_study(sampler=self.sampler, pruner=self.pruner,
            direction=self.direction)
        self.study.optimize(objective, n_trials=self.n_trials)

    def adjust(self, data_trainval=None):
        self.adjust_hparams(data_trainval)
        self.is_training = False
        best_hparams = self.get_hparams()
        self.adjust_params(hparams=best_hparams, data_train=data_trainval)

    def evaluate(self, data=None):

        if self.cross_test:
            fold = self.k_fold_split(self.k_test, data=data)
        else:
            fold = [self.train_test_split(test_size=self.test_size, data=data)]
        results = []
        for data_trainval, data_test in fold:
            self.adjust(data_trainval)
            result = self.eval_param(is_objective=False, data_test=data_test)
            results.append(result)
        if len(results) == 1:
            return results[0]
        else:
            df = pd.DataFrame(index=np.arange(self.k_test), columns=results[0].index)
            for i_fold in range(self.k_test):
                df.loc[i_fold] = results[i_fold]
            return df