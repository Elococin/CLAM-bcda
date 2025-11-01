import os
import torch
import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data import Dataset
import h5py
from utils.utils import generate_split, nth
from pathlib import Path


class Generic_Split:
    """Wrapper for a DataFrame used as a split."""

    def __init__(self, df, data_dir=None, num_classes=None):
        self.slide_data = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.num_classes = num_classes

    def __len__(self):
        """Return number of slides/samples in this split."""
        return len(self.slide_data)
    
    def __getitem__(self, idx):
        """Return (features_tensor, label) tuple for the given index.
        Loads the corresponding .h5 feature file into memory."""
        row = self.slide_data.iloc[idx]
        slide_id = str(row['slide_id'])
        label = int(row['label'])
        if 'feat_dir' in row and isinstance(row['feat_dir'], str) and os.path.exists(row['feat_dir']):
            feature_path = os.path.join(row['feat_dir'], slide_id)
        elif self.data_dir is not None:
            feature_path = os.path.join(self.data_dir, slide_id)
        else:
            feature_path = slide_id
        
        if not feature_path.endswith('.h5'):
            feature_path += '.h5'
        
        if not os.path.exists(feature_path):
            raise FileNotFoundError(
                f"Feature file missing or unreadable: {feature_path}\n"
                f"(from dataset: {row.get('dataset', 'unknown')}, slide_id: {slide_id})"
                )

        with h5py.File(feature_path, 'r') as f:
            features = torch.tensor(f['features'][:], dtype=torch.float32)

        return features, label


def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    if not boolean_style:
        splits = [pd.Series(d.slide_data['slide_id']) for d in split_datasets]
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        index = [sid for d in split_datasets for sid in d.slide_data['slide_id'].tolist()]
        one_hot = np.eye(len(split_datasets), dtype=bool)
        counts = [len(d.slide_data) for d in split_datasets]
        bool_array = np.repeat(one_hot, counts, axis=0)
        df = pd.DataFrame(bool_array, index=index, columns=column_keys)
    df.to_csv(filename, index=False)


class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self, csv_path=None, label_dir=None, shuffle=False, seed=7,
                 print_info=True, label_dict=None, filter_dict=None, ignore=None,
                 patient_strat=False, label_col=None, patient_voting='max'):
        label_dict = label_dict or {}
        filter_dict = filter_dict or {}
        ignore = ignore or []

        self.label_dict = label_dict
        self.num_classes = len(set(self.label_dict.values())) if len(self.label_dict) > 0 else 0
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids = self.val_ids = self.test_ids = None
        self.data_dir = None
        self.label_col = label_col or 'label'
        # remember csv_path for downstream inspection
        self.csv_path = csv_path

        if label_dir is not None:
            all_csvs = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('_ER.csv')]
            assert len(all_csvs) > 0
            dfs = [pd.read_csv(f) for f in all_csvs]
            slide_data = pd.concat(dfs, ignore_index=True)
        else:
            assert csv_path is not None
            slide_data = pd.read_csv(csv_path)

        slide_data = self.filter_df(slide_data, filter_dict)
        slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)
        if shuffle:
            slide_data = slide_data.sample(frac=1, random_state=seed).reset_index(drop=True)
        self.slide_data = slide_data

        self.patient_data_prep(patient_voting)
        self.cls_ids_prep()

        if print_info:
            self.summarize()

    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]
        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self, patient_voting='max'):
        if 'case_id' in self.slide_data.columns:
            patients = np.unique(self.slide_data['case_id'].astype(str))
            cid = 'case_id'
        else:
            patients = np.unique(self.slide_data['slide_id'].astype(str))
            cid = 'slide_id'
        patient_labels = []
        for p in patients:
            locs = self.slide_data[self.slide_data[cid].astype(str) == p].index.tolist()
            assert len(locs) > 0
            labs = self.slide_data['label'][locs].values
            if patient_voting == 'max':
                lab = labs.max()
            elif patient_voting == 'maj':
                lab = stats.mode(labs, keepdims=False)[0]
            else:
                raise NotImplementedError
            patient_labels.append(lab)
        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

    @staticmethod
    def df_prep(data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()
        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            mapped = None
            if key in label_dict:
                mapped = label_dict[key]
            else:
                kstr = str(key)
                if kstr in label_dict:
                    mapped = label_dict[kstr]
                else:
                    try:
                        kint = int(float(key))
                        if kint in label_dict:
                            mapped = label_dict[kint]
                    except Exception:
                        mapped = None
            if mapped is None:
                raise KeyError(f"Label key {key!r} not found in label_dict {label_dict}")
            data.at[i, 'label'] = mapped
        return data

    def filter_df(self, df, filter_dict=None):
        if not filter_dict:
            return df
        mask = np.full(len(df), True, bool)
        for k, v in filter_dict.items():
            mask = np.logical_and(mask, df[k].isin(v))
        return df[mask]

    def __len__(self):
        return len(self.patient_data['case_id']) if self.patient_strat else len(self.slide_data)

    def summarize(self):
        print(f"label column: {self.label_col}")
        print(f"label dictionary: {self.label_dict}")
        print(f"number of classes: {self.num_classes}")
        print('slide-level counts:\n', self.slide_data['label'].value_counts(sort=False))

    def get_split_from_df(self, all_splits, split_key='train'):
        split = all_splits[split_key].dropna().reset_index(drop=True)
        if len(split) > 0:
            if not all(str(s).endswith('.h5') for s in split.tolist()):
                split = split.apply(lambda x: str(x) + '.h5' if not str(x).endswith('.h5') else str(x))

            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            return Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
        return None

    def return_splits(self, from_id=True, csv_path=None):
        if from_id:
            train = Generic_Split(self.slide_data.loc[self.train_ids].reset_index(drop=True), data_dir=self.data_dir, num_classes=self.num_classes) if self.train_ids is not None else None
            val = Generic_Split(self.slide_data.loc[self.val_ids].reset_index(drop=True), data_dir=self.data_dir, num_classes=self.num_classes) if self.val_ids is not None else None
            test = Generic_Split(self.slide_data.loc[self.test_ids].reset_index(drop=True), data_dir=self.data_dir, num_classes=self.num_classes) if self.test_ids is not None else None
            return train, val, test
        assert csv_path
        all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)
        return (self.get_split_from_df(all_splits, 'train'),
                self.get_split_from_df(all_splits, 'val'),
                self.get_split_from_df(all_splits, 'test'))


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self, data_dir, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = int(self.slide_data['label'][idx])
        data_dir = self.data_dir
        # support dict mapping
        if isinstance(self.data_dir, dict):
            if 'source' in self.slide_data.columns:
                src = self.slide_data['source'][idx]
                data_dir = self.data_dir.get(src, None)
            else:
                data_dir = list(self.data_dir.values())[0] if len(self.data_dir) > 0 else None

        # check pt_files
        if data_dir:
            pt_path = os.path.join(data_dir, 'pt_files', f'{slide_id}.pt')
            if os.path.exists(pt_path):
                feats = torch.load(pt_path)
                return feats, label
            h5_can = os.path.join(data_dir, 'h5_files', f'{slide_id}.h5')
            if os.path.exists(h5_can):
                with h5py.File(h5_can, 'r') as hf:
                    arr = hf['features'][()] if 'features' in hf else hf[list(hf.keys())[0]][()]
                    coords = hf['coords'][()] if 'coords' in hf else None
                feats = torch.from_numpy(arr).float()
                if coords is not None:
                    try:
                        return feats, label, torch.from_numpy(coords)
                    except Exception:
                        return feats, label
                return feats, label

        # heuristic search under candidate roots
        candidates = []
        if data_dir:
            candidates.append(Path(data_dir))
            candidates.append(Path(data_dir).parent)
        subdirs = ['bcnb', 'haiti', 'postnat_brca', 'tcga_brca']
        found = None
        for root in candidates:
            if not root or not root.exists():
                continue
            for sub in subdirs:
                p = root / sub
                if not p.exists() or not p.is_dir():
                    continue
                exact = p / f'{slide_id}.h5'
                if exact.exists():
                    found = exact
                    break
                for f in p.iterdir():
                    if f.is_file() and f.suffix.lower() == '.h5' and slide_id.lower() in f.name.lower():
                        found = f
                        break
                if found:
                    break
            if found:
                break

        if not found and data_dir:
            for f in Path(data_dir).rglob('*.h5'):
                if slide_id.lower() in f.name.lower():
                    found = f
                    break

        if not found:
            searched = [str(p) for p in candidates if p]
            raise FileNotFoundError(f"No feature file for slide_id={slide_id}. Searched: {searched}")

        with h5py.File(found, 'r') as hf:
            arr = hf['features'][()] if 'features' in hf else hf[list(hf.keys())[0]][()]
            coords = hf['coords'][()] if 'coords' in hf else None
        feats = torch.from_numpy(arr).float()
        if coords is not None:
            try:
                return feats, label, torch.from_numpy(coords)
            except Exception:
                return feats, label
        return feats, label

    def __len__(self):
        return len(self.slide_data)
