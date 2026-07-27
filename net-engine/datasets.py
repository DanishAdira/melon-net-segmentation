import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import numpy as np
from pathlib import Path
from ultralytics import YOLO

ImageFile.LOAD_TRUNCATED_IMAGES = True

class MelonDataset(Dataset):
    def __init__(self, data_dir, list_file_name, texture_map_folder_name=None, transform=None):
        """
        Args:
            data_dir (str): データセットが格納されているディレクトリのパス。
            list_file (str): 画像ファイル名を記載したテキストファイルのパス（train.txt または validation.txt）。
            texture_map_folder_name (str): テクスチャマップが格納されているディレクトリの名前。
            transform (callable, optional): 画像データに適用する変換（オーグメンテーション）。
        """
        self.data_dir = [data_dir]
        self.img_dirs = [os.path.join(data_dir, 'images')]
        self.mask_dirs = [os.path.join(data_dir, 'masks')]
        if texture_map_folder_name:
            self.texture_map_dirs = [os.path.join(data_dir, texture_map_folder_name)]
        else:
            self.texture_map_dirs = None
        self.transform = transform
        self.list_file_name = [list_file_name]
        self.image_names = self._get_image_names_from_list_file()

    def _get_image_names_from_list_file(self):
        """テキストファイルを読み込み、画像ファイル名のリストを作成"""
        list_file_path = os.path.join(self.data_dir[0], self.list_file_name[0])
        with open(list_file_path) as f:
            image_names = [line.strip() for line in f.readlines()]
            return image_names
        raise FileNotFoundError(f"File not found: {list_file_path}")

    def _get_img_path(self, folders, idx):
        """指定されたフォルダ群から画像のパスを取得"""
        for folder in folders:
            img_path = Path(folder) / Path(self.image_names[idx]).name
            # 拡張子がついている場合はそのまま追加
            if Path(img_path).suffix:
                if Path(img_path).exists():
                    return img_path
            # 拡張子がついていない場合は拡張子を追加して検索
            else:
                serch_ext = ['.jpg', '.jpeg', '.png']
                for ext in serch_ext:
                    if Path(img_path).with_suffix(ext).exists():
                        return str(Path(img_path).with_suffix(ext))
        raise FileNotFoundError(f"File not found: {img_path}")

    def __len__(self):
        """データセットのサンプル数を返す"""
        return len(self.image_names)

    def __add__(self, other):
        self.img_dirs.extend(other.img_dirs)
        self.img_dirs.extend(other.img_dirs)
        self.mask_dirs.extend(other.mask_dirs)
        if self.texture_map_dirs:
            self.texture_map_dirs.extend(other.texture_map_dirs)
        self.list_file_name.extend(other.list_file_name)
        self.image_names.extend(other.image_names)
        return self

    def __iadd__(self, other):
        return self.__add__(other)

    def __getitem__(self, idx):
        """指定されたインデックスの画像と対応するマスクを返す"""
        img_path = self._get_img_path(self.img_dirs, idx)
        mask_path = self._get_img_path(self.mask_dirs, idx)

        # 画像とGT画像(マスク画像)を読み込む
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"))
        except OSError:
            print(f"Failed to load image/mask at: {img_path}")
            return None

        # マスク画像のピクセル値を0または1に変換
        mask = np.where(mask > 128, 1, 0).astype(np.float32)

        # 画像とテクスチャマップをチャネル方向に結合
        if self.texture_map_dirs:
            texture_map_path = self._get_img_path(self.texture_map_dirs, idx)
            texture_map = np.array(Image.open(texture_map_path).convert("L"))
            texture_map = np.expand_dims(texture_map, axis=-1)
            image= np.concatenate([image, texture_map], axis=-1)

        # 画像とマスクに変換を適用
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask