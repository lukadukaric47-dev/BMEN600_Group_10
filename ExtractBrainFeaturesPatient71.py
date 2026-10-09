from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import (
    label,
    binary_fill_holes,
    binary_closing,
    binary_opening,
    binary_erosion
)


# ============================================================
# 1. DATASET PATH
# ============================================================

dataset_root = Path(
    r"C:\Users\milad\OneDrive - University of Calgary\Courses\Fall 2026\BMEN-600-Biomedical-Engineering-Foundation\files-20261006T174136Z-1-001\files\ct-ich\1.3.1"
)

ct_folder = dataset_root / "ct_scans"


# ============================================================
# 2. PATIENT
# ============================================================

patient_id = 71


# ============================================================
# 3. FIND CT FILE
# ============================================================

possible_ct_files = [
    ct_folder / f"{patient_id:03d}.nii",
    ct_folder / f"{patient_id:03d}.nii.gz",
    ct_folder / f"{patient_id}.nii",
    ct_folder / f"{patient_id}.nii.gz",
]

ct_path = None

for file_path in possible_ct_files:

    if file_path.exists():

        ct_path = file_path
        break


if ct_path is None:

    raise FileNotFoundError(
        f"Could not find CT file for patient {patient_id}"
    )


# ============================================================
# 4. LOAD CT
# ============================================================

ct_image = nib.load(
    str(ct_path)
)

ct_data = ct_image.get_fdata()

voxel_size = ct_image.header.get_zooms()[:3]


print("========================================")
print("CT INFORMATION")
print("========================================")

print(f"Patient ID: {patient_id}")
print(f"CT file: {ct_path.name}")
print(f"Shape: {ct_data.shape}")
print(f"Voxel size: {voxel_size}")

print()


# ============================================================
# 5. HELPER FUNCTION:
#    KEEP LARGE CONNECTED COMPONENTS
# ============================================================

def keep_large_components(binary_image, minimum_size=200):

    labeled_image, number_of_components = label(
        binary_image
    )

    if number_of_components == 0:

        return np.zeros_like(
            binary_image,
            dtype=bool
        )


    component_sizes = np.bincount(
        labeled_image.ravel()
    )


    keep_component = (
        component_sizes >= minimum_size
    )


    keep_component[0] = False


    cleaned_image = keep_component[
        labeled_image
    ]


    return cleaned_image


# ============================================================
# 6. FUNCTION TO CREATE BRAIN MASK FOR ONE SLICE
# ============================================================

def create_brain_mask(ct_slice):

    # --------------------------------------------------------
    # Bone threshold
    # --------------------------------------------------------

    bone_mask = (
        ct_slice >= 200
    )


    # --------------------------------------------------------
    # Close gaps in skull
    # --------------------------------------------------------

    closed_bone = binary_closing(
        bone_mask,
        structure=np.ones(
            (5, 5),
            dtype=bool
        ),
        iterations=2
    )


    # --------------------------------------------------------
    # Fill inside skull
    # --------------------------------------------------------

    skull_filled = binary_fill_holes(
        closed_bone
    )


    height, width = ct_slice.shape

    center_y = height // 2
    center_x = width // 2


    center_is_inside = bool(
        skull_filled[
            center_y,
            center_x
        ]
    )


    filled_fraction = (
        np.sum(skull_filled)
        /
        skull_filled.size
    )


    # --------------------------------------------------------
    # If skull filling is unreliable, use head-mask fallback
    # --------------------------------------------------------

    if (
        not center_is_inside
        or
        filled_fraction < 0.03
    ):

        head_mask = (
            ct_slice > -200
        )


        head_mask = keep_large_components(
            head_mask,
            minimum_size=1000
        )


        head_mask = binary_fill_holes(
            head_mask
        )


        intracranial_mask = binary_erosion(
            head_mask,
            structure=np.ones(
                (3, 3),
                dtype=bool
            ),
            iterations=10
        )


    else:

        intracranial_mask = binary_erosion(
            skull_filled,
            structure=np.ones(
                (3, 3),
                dtype=bool
            ),
            iterations=3
        )


    # --------------------------------------------------------
    # Plausible intracranial soft-tissue HU range
    # --------------------------------------------------------

    brain_intensity_mask = (
        (ct_slice >= -20)
        &
        (ct_slice <= 120)
    )


    # --------------------------------------------------------
    # Combine geometry and HU
    # --------------------------------------------------------

    brain_mask = (
        intracranial_mask
        &
        brain_intensity_mask
    )


    # --------------------------------------------------------
    # Clean small noise
    # --------------------------------------------------------

    brain_mask = binary_opening(
        brain_mask,
        structure=np.ones(
            (3, 3),
            dtype=bool
        ),
        iterations=1
    )


    brain_mask = binary_closing(
        brain_mask,
        structure=np.ones(
            (3, 3),
            dtype=bool
        ),
        iterations=1
    )


    brain_mask = keep_large_components(
        brain_mask,
        minimum_size=200
    )


    return brain_mask


# ============================================================
# 7. CREATE 3D BRAIN MASK
# ============================================================

brain_mask_3d = np.zeros_like(
    ct_data,
    dtype=bool
)


print("========================================")
print("CREATING 3D BRAIN MASK")
print("========================================")


for slice_index in range(
    ct_data.shape[2]
):

    ct_slice = ct_data[
        :,
        :,
        slice_index
    ]


    brain_mask = create_brain_mask(
        ct_slice
    )


    brain_mask_3d[
        :,
        :,
        slice_index
    ] = brain_mask


    brain_pixels = int(
        np.sum(brain_mask)
    )


    print(
        f"Slice {slice_index:02d}: "
        f"{brain_pixels} brain pixels"
    )


print()


# ============================================================
# 8. TOTAL BRAIN VOXELS
# ============================================================

brain_voxels = int(
    np.sum(
        brain_mask_3d
    )
)


print("========================================")
print("3D BRAIN MASK")
print("========================================")

print(
    f"Total brain voxels: "
    f"{brain_voxels}"
)


# ============================================================
# 9. BRAIN VOLUME
# ============================================================

voxel_volume_mm3 = (
    float(voxel_size[0])
    *
    float(voxel_size[1])
    *
    float(voxel_size[2])
)


brain_volume_mm3 = (
    brain_voxels
    *
    voxel_volume_mm3
)


brain_volume_ml = (
    brain_volume_mm3
    /
    1000.0
)


print(
    f"Approximate brain volume: "
    f"{brain_volume_ml:.2f} mL"
)

print()


# ============================================================
# 10. WHOLE-BRAIN HU VALUES
# ============================================================

brain_hu = ct_data[
    brain_mask_3d
]


brain_hu = brain_hu[
    np.isfinite(
        brain_hu
    )
]


mean_brain_hu = float(
    np.mean(
        brain_hu
    )
)


median_brain_hu = float(
    np.median(
        brain_hu
    )
)


std_brain_hu = float(
    np.std(
        brain_hu
    )
)


print("========================================")
print("WHOLE-BRAIN HU")
print("========================================")

print(
    f"Mean brain HU: "
    f"{mean_brain_hu:.2f}"
)

print(
    f"Median brain HU: "
    f"{median_brain_hu:.2f}"
)

print(
    f"Brain HU standard deviation: "
    f"{std_brain_hu:.2f}"
)

print()


# ============================================================
# 11. LEFT / RIGHT BRAIN MASKS
# ============================================================

left_mask_3d = np.zeros_like(
    brain_mask_3d,
    dtype=bool
)

right_mask_3d = np.zeros_like(
    brain_mask_3d,
    dtype=bool
)


# ============================================================
# 12. SPLIT EACH SLICE USING BRAIN CENTER
# ============================================================

for slice_index in range(
    ct_data.shape[2]
):

    brain_mask = brain_mask_3d[
        :,
        :,
        slice_index
    ]


    coordinates = np.argwhere(
        brain_mask
    )


    # Skip empty or extremely small slices
    if len(coordinates) < 200:

        continue


    # --------------------------------------------------------
    # Estimate center of the brain on this slice
    # --------------------------------------------------------

    center_x = float(
        np.median(
            coordinates[:, 0]
        )
    )


    x_coordinates = np.arange(
        brain_mask.shape[0]
    )[:, None]


    # --------------------------------------------------------
    # Split brain mask into two sides
    # --------------------------------------------------------

    left_slice_mask = (
        brain_mask
        &
        (
            x_coordinates
            <
            center_x
        )
    )


    right_slice_mask = (
        brain_mask
        &
        (
            x_coordinates
            >=
            center_x
        )
    )


    left_mask_3d[
        :,
        :,
        slice_index
    ] = left_slice_mask


    right_mask_3d[
        :,
        :,
        slice_index
    ] = right_slice_mask


# ============================================================
# 13. EXTRACT LEFT / RIGHT HU VALUES
# ============================================================

left_hu = ct_data[
    left_mask_3d
]


right_hu = ct_data[
    right_mask_3d
]


left_hu = left_hu[
    np.isfinite(
        left_hu
    )
]


right_hu = right_hu[
    np.isfinite(
        right_hu
    )
]


# ============================================================
# 14. CALCULATE LEFT / RIGHT MEAN HU
# ============================================================

mean_left_hu = float(
    np.mean(
        left_hu
    )
)


mean_right_hu = float(
    np.mean(
        right_hu
    )
)


print("========================================")
print("LEFT / RIGHT HU")
print("========================================")

print(
    f"Mean left-side HU: "
    f"{mean_left_hu:.2f}"
)

print(
    f"Mean right-side HU: "
    f"{mean_right_hu:.2f}"
)

print()


# ============================================================
# 15. ASYMMETRY INDEX
# ============================================================

denominator = (
    mean_left_hu
    +
    mean_right_hu
)


if denominator != 0:

    asymmetry_index = (
        200
        *
        abs(
            mean_left_hu
            -
            mean_right_hu
        )
        /
        denominator
    )

else:

    asymmetry_index = np.nan


print("========================================")
print("ASYMMETRY INDEX")
print("========================================")

print(
    f"Asymmetry Index: "
    f"{asymmetry_index:.2f}%"
)

print()


# ============================================================
# 16. FIND REPRESENTATIVE SLICE
# ============================================================

brain_voxels_per_slice = np.sum(
    brain_mask_3d,
    axis=(0, 1)
)


representative_slice = int(
    np.argmax(
        brain_voxels_per_slice
    )
)


print(
    f"Representative slice: "
    f"{representative_slice}"
)

print()


# ============================================================
# 17. PREPARE REPRESENTATIVE SLICE
# ============================================================

ct_slice = ct_data[
    :,
    :,
    representative_slice
]


brain_slice = brain_mask_3d[
    :,
    :,
    representative_slice
]


left_slice = left_mask_3d[
    :,
    :,
    representative_slice
]


right_slice = right_mask_3d[
    :,
    :,
    representative_slice
]


# ============================================================
# 18. BRAIN WINDOW
# ============================================================

window_center = 40
window_width = 80


lower_limit = (
    window_center
    -
    window_width / 2
)


upper_limit = (
    window_center
    +
    window_width / 2
)


windowed_ct = np.clip(
    ct_slice,
    lower_limit,
    upper_limit
)


# ============================================================
# 19. DISPLAY RESULTS
# ============================================================

plt.figure(
    figsize=(16, 4)
)


# ------------------------------------------------------------
# Original CT
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    1
)

plt.imshow(
    np.rot90(
        windowed_ct
    ),
    cmap="gray",
    vmin=lower_limit,
    vmax=upper_limit
)

plt.title(
    f"Original CT\n"
    f"Slice {representative_slice}"
)

plt.axis(
    "off"
)


# ------------------------------------------------------------
# Brain mask
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    2
)

plt.imshow(
    np.rot90(
        brain_slice
    ),
    cmap="gray"
)

plt.title(
    "3D Brain Mask"
)

plt.axis(
    "off"
)


# ------------------------------------------------------------
# Brain mask overlay
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    3
)

plt.imshow(
    np.rot90(
        windowed_ct
    ),
    cmap="gray",
    vmin=lower_limit,
    vmax=upper_limit
)

plt.imshow(
    np.rot90(
        brain_slice
    ),
    cmap="Greens",
    alpha=0.30
)

plt.title(
    "CT + Brain Mask"
)

plt.axis(
    "off"
)


# ------------------------------------------------------------
# Left / right split
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    4
)

plt.imshow(
    np.rot90(
        windowed_ct
    ),
    cmap="gray",
    vmin=lower_limit,
    vmax=upper_limit
)

plt.imshow(
    np.rot90(
        left_slice
    ),
    cmap="Blues",
    alpha=0.35
)

plt.imshow(
    np.rot90(
        right_slice
    ),
    cmap="Reds",
    alpha=0.35
)

plt.title(
    "Left / Right Split"
)

plt.axis(
    "off"
)


plt.tight_layout()

plt.show()