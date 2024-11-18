import numpy as np
import pandas as pd
import scanpy as sc
import time
import os
from pathlib import Path

from PIL.Image import NONE
import cv2
from anndata import AnnData

from typing import Optional, Union
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from matplotlib.image import imread


workPa    = '/sibcb2/chenluonanlab7/cmzuo/workPath/Spatial_EQTL/Breast_cancer/'
slice_set = ['BRCA_InvasiveDuctalCarcinoma_StainedWithFluorescent_CD3Antibody_10x', 'BRCA_BlockASection1_10x']
savePath  = '/sibcb1/chenluonanlab8/cmzuo/workPath/stClinic/BRCA/IDC_BAS1/'
sli_ap    = ['IDC_CONCH', 'BAS1_CONCH']

for zz in list(range(len(slice_set))):
	print(slice_set[zz])

	input_dir  = os.path.join(workPa, slice_set[zz])

	adata      = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5', load_images=True)
	library_id = list(adata.uns["spatial"].keys())[0]
	scale      = adata.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"]
	image_coor = adata.obsm["spatial"] * scale
	adata.obs["imagecol"] = image_coor[:, 0]
	adata.obs["imagerow"] = image_coor[:, 1]
	adata.uns["spatial"][library_id]["use_quality"] = 'hires'

	tillingPath = savePath + sli_ap[zz] +'/tmp/'
	tiling(adata, tillingPath)


def tiling(
	adata: AnnData,
	out_path: str = None,
	library_id: str = None,
	crop_size: int = 40,
	target_size: int = 32,
	verbose: bool = False,
	copy: bool = False,
) -> Optional[AnnData]:
	"""
	adopted from stLearn package
	Tiling H&E images to small tiles based on spot spatial location
	"""

	if library_id is None:
		library_id = list(adata.uns["spatial"].keys())[0]

	# Check the exist of out_path
	if not os.path.isdir(out_path):
		os.mkdir(out_path)

	image = adata.uns["spatial"][library_id]["images"][
		adata.uns["spatial"][library_id]["use_quality"]
	]
	if image.dtype == np.float32 or image.dtype == np.float64:
		image = (image * 255).astype(np.uint8)
	img_pillow = Image.fromarray(image)
	tile_names = []

	with tqdm(
		total=len(adata),
		desc="Tiling image",
		bar_format="{l_bar}{bar} [ time left: {remaining} ]",
	) as pbar:
		for barcode, imagerow, imagecol in zip(adata.obs.index, adata.obs["imagerow"], adata.obs["imagecol"]):
			imagerow_down  = imagerow - crop_size / 2
			imagerow_up    = imagerow + crop_size / 2
			imagecol_left  = imagecol - crop_size / 2
			imagecol_right = imagecol + crop_size / 2
			tile           = img_pillow.crop( (imagecol_left, imagerow_down,
											   imagecol_right, imagerow_up)
											)
			tile.thumbnail((target_size, target_size), Image.ANTIALIAS)
			#tile.resize((target_size, target_size))

			tile_name = str(barcode)
			out_tile  = Path(out_path) / (tile_name + ".jpeg")

			tile_names.append(str(out_tile))

			if verbose:
				print(
					"generate tile at location ({}, {})".format(
						str(imagecol), str(imagerow)
					)
				)
			tile.save(out_tile, "JPEG")

			pbar.update(1)

	adata.obs["tile_path"] = tile_names
	return adata if copy else None


