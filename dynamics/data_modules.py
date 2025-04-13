"""
A data module that will separate the data into different folds. This data module
will also be able to train on full sequences

Each fold is separated by contiguous shot numbers.
"""

from typing import Dict, Union, List, Sequence, Tuple, Optional
import os
import pickle as pkl

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from dynamics.utils import get_shots
from dynamics_toolbox.utils.storage.qdata import load_from_hdf5


class KFoldSequenceFusionDataModule(LightningDataModule):

    def __init__(
            self,
            data_path: str,
            batch_size: int,
            shot_train_length: int,
            min_shot_amt: int,
            n_folds: int,
            te_fold: int,
            save_dir: str,
            file_name: str = 'full.hdf5',
            prop_validation: float = 0.1,
            num_workers: int = 1,
            pin_memory: bool = True,
            seed: int = 1,
            bootstrapped: bool = True,
            train_from_start_only: bool = True
    ):
        """Constructor.
        Args:
            data_path: Path to the directory containing the shot data.
            batch_size: Batch size.
            shot_train_length: The length of the shot to train on. Padding will be
                added so that all batches drawn are this size.
            min_shot_amt: The minimum amount of data in a shot to consider using it.
            n_folds: The number of folds to separate the data into. If the number of
                folds is 1. Then no test fold will be allocated.
            te_fold: The fold to hold out for testing, can take values {1, ..., n_folds}
            save_dir: where to save the shot numbers used for training/testing/valiation
            file_name: The file name of the data that should be loaded in.
            prop_validation: The proporton of the total data to be used for validation.
                This is important for preventing overfitting.
            num_workers: Number of workers.
            pin_memory: Whether to pin memory.
            seed: The seed. The randomness here is the selection of the validation shots.
            bootstrapped: Whether to bootstrap the raw dataset.
            train_from_start_only: Whether to only train starting from the start of
                the shot or whether to have a moving window we can use for several
                different Sequence snippets.
        """
        
        if n_folds > 1 and (te_fold < 1 or te_fold > n_folds):
            raise ValueError(f'te_fold must be either 1, ..., {n_folds}')
        
        super().__init__()
        np.random.seed(seed)
        self.shot_train_length = shot_train_length
        self.min_shot_amt = min_shot_amt
        self.train_from_start_only = train_from_start_only
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._bootstrapped = bootstrapped

        # not used for now
        # with open(os.path.join(data_path, 'info.pkl'), 'rb') as f:
            # self.info = pkl.load(f)

        # state_idxs=[]
        # for state in remove_states:
        #     state_idxs.append(self.info['state_space'].index(state))
        # data['states'] = np.delete(data['states'], state_idxs, axis=1)
        # data['next_states'] = np.delete(data['next_states'], state_idxs, axis=1)

        # actuator_idxs=[]
        # for actuator in remove_actuators:
        #     actuator_idxs.append(self.info['actuator_space'].index(actuator))
        # data['actuators'] = np.delete(data['actuators'], actuator_idxs, axis=1)
        # data['next_actuators'] = np.delete(data['next_actuators'], actuator_idxs, axis=1)

        # get the raw/shot datasets
        data = load_from_hdf5(os.path.join(data_path, file_name))
        shot_data = get_shots(data, min_amt_needed=self.min_shot_amt)
        self.te_shot_nums = self._get_test_shots(data, n_folds, te_fold) # The test shots for each ensemble member are the same.

        # TODO: use a better design for bootstrapping sequential data
        # We only bootstrap the training data.
        if self._bootstrapped: 
            # split the shot data
            test_shot_data, other_shot_data = [], []
            for shot in shot_data:
                if shot['shotnum'][0] in self.te_shot_nums:
                    test_shot_data.append(shot)
                else:
                    other_shot_data.append(shot)

            # count the length of each non-test shot
            durations = []
            for shot in other_shot_data:
                durations.append(len(shot['shotnum']))
            durations = np.array(durations)
            # get sampling weights proportional to duration
            probs = durations / durations.sum() # TODO: use this or not
            # bootstrap sampling 
            bootstrap_sample = np.random.choice(other_shot_data, size=len(other_shot_data), replace=True, p=probs)
            shot_data = test_shot_data + list(bootstrap_sample)

        # get the training/validation shots
        self.tr_shot_nums, self.val_shot_nums = self._separate_shot_nums(shot_data, prop_validation) 
        
        self.tr_set, self.val_set, self.te_set = self._assemble_datasets(
            shot_data, [self.tr_shot_nums, self.val_shot_nums, self.te_shot_nums]
        )

        for name, dset in (('tr', self.tr_set),
                           ('val', self.val_set),
                           ('te', self.te_set)):
            setattr(self, f'_{name}_dataset', TensorDataset(
                torch.Tensor(dset[0]), # input
                torch.Tensor(dset[1]), # output
                torch.Tensor(dset[2]), # padding mask
            ))

        # summary
        print('Data Loaded in!')
        print(f'\t Number training shots: {len(self.tr_shot_nums)}')
        print(f'\t Number validation shots: {len(self.val_shot_nums)}')
        print(f'\t Number test shots: {len(self.te_shot_nums)}')

        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'tr_shots.npy'), np.array([si for si in self.tr_shot_nums]))
        np.save(os.path.join(save_dir, 'val_shots.npy'), np.array([si for si in self.val_shot_nums]))
        np.save(os.path.join(save_dir, 'te_shots.npy'), np.array([si for si in self.te_shot_nums]))

    def train_dataloader(self) ->\
            Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """Get the training dataloader."""
        return DataLoader(
            self._tr_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=self._pin_memory,
        )

    def val_dataloader(self) ->\
            Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """Get the training dataloader."""
        if len(self._val_dataset):
            return DataLoader(
                self._val_dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=self._pin_memory,
            )
        else:
            None

    def test_dataloader(self) ->\
            Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """Get the training dataloader."""
        if len(self._te_dataset):
            return DataLoader(
                self._te_dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=self._pin_memory,
            )
        else:
            None

    def _get_test_shots(self, data, n_folds, te_fold):
        """
        Args:
            data: The full dataset loaded in.
            n_folds: The number of folds to separate the data into.
            te_fold: The fold to hold out for testing, can take values {1, ..., n_folds}
        Returns:
            The test shots.
        """
        shots = np.sort(list(set(data['shotnum']))) 

        if n_folds == 1:  # In this case there are no test shots.
            te_shots = set([])
        else:
            shots_per_fold = len(shots) // n_folds
            lidx = int(shots_per_fold * (te_fold - 1))
            ridx = int(shots_per_fold * te_fold)
            te_shots = set(shots[lidx:ridx])
        
        return te_shots

    def _separate_shot_nums(
            self,
            shot_data,
            prop_validation: float
    ) -> Sequence[set]:
        """Separate the remaining shots into a train and validation set.

        Args:
            shot_data: The full shot dataset.
            val_shot_nums: Shot numbers that should be included in the validation set.

        Returns:
            The train and validation shots.
        """

        remaining_shots = []
        for shot in shot_data:
            if shot["shotnum"][0] not in self.te_shot_nums:
                remaining_shots.append(shot["shotnum"][0])
        remaining_shots = np.array(list(set(remaining_shots)))
        num_val = int(prop_validation * len(remaining_shots))

        if num_val <= 0:
            tr_shots = set(remaining_shots)
            val_shots = set([])
        else:
            val_idxs = np.random.choice(len(remaining_shots), size=num_val, replace=False)
            val_shots = set([rs for rs in remaining_shots[val_idxs]])
            tr_shots = set([rs for rs in remaining_shots if rs not in val_shots])

        return tr_shots, val_shots

    def _assemble_datasets(
            self,
            shot_data,
            shot_nums: Sequence[set],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assembe the datasets.

        Args:
            shot_data: The shot dataset.
            shot_nums: The training, validation, and test set of shots.

        Returns:
            The training, validation, and test set of
            (input, output, padding)
        """

        tr, val, te = [[[] for _ in range(3)] for _ in range(3)]

        for sdata in shot_data:
            snum = sdata['shotnum'][0]
            if snum in shot_nums[0]:
                to_add = tr
            elif snum in shot_nums[1]:
                to_add = val
            else:
                to_add = te

            if (self.train_from_start_only or len(sdata['observations']) <= self.shot_train_length):
                pad_needed = max(self.shot_train_length - len(sdata['observations']), 0)
                # Add the x data.
                to_add[0].append(np.concatenate([
                    np.concatenate([
                        sdata['observations'],
                        sdata['pre_actions'],
                        sdata['action_deltas'],
                    ], axis=-1),
                    np.zeros((pad_needed,
                              sdata['observations'].shape[1]
                              + sdata['pre_actions'].shape[1]
                              + sdata['action_deltas'].shape[1])),
                ], axis=0)[:self.shot_train_length])
                # Add the y data.
                to_add[1].append(np.concatenate([
                    sdata['observation_deltas'],
                    np.zeros((pad_needed, sdata['observation_deltas'].shape[1]))], axis=0)[:self.shot_train_length])
                # Add the masks.
                to_add[-1].append(np.append(
                    np.ones((min(len(sdata['observations']), self.shot_train_length), 1)),
                    np.zeros((pad_needed, 1))))
                
            else:
                num_snippets = len(sdata['observations']) - self.shot_train_length
                for ns in range(num_snippets):
                    # Add the x data.
                    to_add[0].append(np.concatenate([
                        sdata['observations'][ns:ns+self.shot_train_length],
                        sdata['pre_actions'][ns:ns+self.shot_train_length],
                        sdata['action_deltas'][ns:ns+self.shot_train_length],
                    ], axis=-1))
                    # Add the y data.
                    to_add[1].append(sdata['observation_deltas'][ns:ns+self.shot_train_length])
                    to_add[-1].append(np.ones((self.shot_train_length, 1)))

        for dt in [tr, val, te]:
            for didx, dlist in enumerate(dt):
                dt[didx] = np.array(dlist)
                if len(dt[didx].shape) < 3:
                    dt[didx] = dt[didx][..., np.newaxis] # (70, 225, 55), (shot_num, shot_length, datapoint_dim)

        return tr, val, te

    @property
    def data(self) -> Sequence[np.array]:
        """Get all of the data."""
        return self.tr_set[:2]

    @property
    def input_data(self) -> np.array:
        """The input data.."""
        return self.tr_set[0]

    @property
    def output_data(self) -> np.array:
        """The output data."""
        return self.tr_set[1]

    @property
    def input_dim(self) -> int:
        """Observation dimension."""
        return int(self.tr_set[0].shape[-1])

    @property
    def output_dim(self) -> int:
        return int(self.tr_set[1].shape[-1])

    @property
    def num_train(self) -> int:
        """Number of training points."""
        return len(self.tr_set[0])

    @property
    def num_validation(self) -> int:
        """Number of training points."""
        return len(self.val_set[0])

    @property
    def num_test(self) -> int:
        """Number of training points."""
        return len(self.te_set[0])