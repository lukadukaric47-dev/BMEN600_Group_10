from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


# ============================================================
# 1. DATASET ROOT
# ============================================================

dataset_root = Path(
    r"C:\Users\milad\OneDrive - University of Calgary\Courses\Fall 2026\BMEN-600-Biomedical-Engineering-Foundation\files-20261006T174136Z-1-001\files\ct-ich\1.3.1"
)


# ============================================================
# 2. INPUT PATHS
# ============================================================

ct_folder = dataset_root / "ct_scans"

mask_folder = dataset_root / "masks"

demographics_file = dataset_root / "Patient_demographics.csv"


# ============================================================
# 3. OUTPUT FILE
# ============================================================

output_file = dataset_root / "patient_table_with_features.csv"


# ============================================================
# 4. CHECK REQUIRED INPUTS
# ============================================================

if not ct_folder.exists():
    raise FileNotFoundError(
        f"CT folder not found:\n{ct_folder}"
    )

if not mask_folder.exists():
    raise FileNotFoundError(
        f"Mask folder not found:\n{mask_folder}"
    )

if not demographics_file.exists():
    raise FileNotFoundError(
        f"Patient demographics file not found:\n{demographics_file}"
    )


# ============================================================
# 5. LOAD PATIENT DEMOGRAPHICS
# ============================================================

# The original CSV has two header rows.
# We skip them and assign clean column names ourselves.

column_names = [
    "Patient_ID",
    "Age_years",
    "Gender",
    "Intraventricular",
    "Intraparenchymal",
    "Subarachnoid",
    "Epidural",
    "Subdural",
    "Fracture",
    "Note"
]


patients = pd.read_csv(
    demographics_file,
    skiprows=2,
    header=None,
    names=column_names
)


# ============================================================
# 6. CLEAN PATIENT ID
# ============================================================

patients["Patient_ID"] = pd.to_numeric(
    patients["Patient_ID"],
    errors="coerce"
)


# Remove summary rows and invalid rows
patients = patients.dropna(
    subset=["Patient_ID"]
).copy()


patients["Patient_ID"] = (
    patients["Patient_ID"].astype(int)
)


# ============================================================
# 7. CLEAN AGE
# ============================================================

patients["Age_years"] = pd.to_numeric(
    patients["Age_years"],
    errors="coerce"
)


# ============================================================
# 8. CLEAN HEMORRHAGE COLUMNS
# ============================================================

hemorrhage_columns = [
    "Intraventricular",
    "Intraparenchymal",
    "Subarachnoid",
    "Epidural",
    "Subdural"
]


for column in hemorrhage_columns:

    patients[column] = pd.to_numeric(
        patients[column],
        errors="coerce"
    )

    patients[column] = (
        patients[column]
        .fillna(0)
        .astype(int)
    )


# ============================================================
# 9. CREATE OVERALL ICH LABEL
# ============================================================

patients["ICH"] = (
    patients[hemorrhage_columns]
    .max(axis=1)
    .astype(int)
)


# ============================================================
# 10. CLEAN FRACTURE COLUMN
# ============================================================

patients["Fracture"] = pd.to_numeric(
    patients["Fracture"],
    errors="coerce"
)

patients["Fracture"] = (
    patients["Fracture"]
    .fillna(0)
    .astype(int)
)


# ============================================================
# 11. AGE QUALITY CONTROL
# ============================================================

def age_quality_control(age):

    if pd.isna(age):
        return "MISSING"

    if age < 1:
        return "CHECK_INFANT_AGE"

    return "OK"


patients["Age_QC"] = (
    patients["Age_years"]
    .apply(age_quality_control)
)


# ============================================================
# 12. SEX BINARY ENCODING
# ============================================================

def encode_sex(value):

    value = str(value).strip().lower()

    if value == "male":
        return 0

    if value == "female":
        return 1

    return np.nan


patients["Sex_binary"] = (
    patients["Gender"]
    .apply(encode_sex)
)


# ============================================================
# 13. FUNCTION TO FIND NIFTI FILE
# ============================================================

def find_nifti_file(folder, patient_id):

    possible_files = [
        folder / f"{patient_id:03d}.nii",
        folder / f"{patient_id:03d}.nii.gz",
        folder / f"{patient_id}.nii",
        folder / f"{patient_id}.nii.gz"
    ]

    for file_path in possible_files:

        if file_path.exists():
            return file_path

    return None


# ============================================================
# 14. FIND WHICH PATIENTS ACTUALLY HAVE CT FILES
# ============================================================

available_patient_ids = []


for patient_id in patients["Patient_ID"]:

    ct_path = find_nifti_file(
        ct_folder,
        patient_id
    )

    if ct_path is not None:
        available_patient_ids.append(
            patient_id
        )


# Keep only patients present in our CT subset
patients = patients[
    patients["Patient_ID"].isin(
        available_patient_ids
    )
].copy()


patients = patients.reset_index(
    drop=True
)


print("========================================")
print("PATIENT DATASET")
print("========================================")

print(
    f"Patients with CT scans: "
    f"{len(patients)}"
)

print()


# ============================================================
# 15. FEATURE EXTRACTION RESULTS
# ============================================================

feature_results = []


# ============================================================
# 16. PROCESS EACH PATIENT
# ============================================================

for _, row in patients.iterrows():

    patient_id = int(
        row["Patient_ID"]
    )


    print("========================================")
    print(f"PATIENT {patient_id}")
    print("========================================")


    # --------------------------------------------------------
    # Find CT
    # --------------------------------------------------------

    ct_path = find_nifti_file(
        ct_folder,
        patient_id
    )


    # --------------------------------------------------------
    # Find mask
    # --------------------------------------------------------

    mask_path = find_nifti_file(
        mask_folder,
        patient_id
    )


    ct_found = int(
        ct_path is not None
    )

    mask_found = int(
        mask_path is not None
    )


    print(
        f"CT:   "
        f"{ct_path.name if ct_path else 'NOT FOUND'}"
    )

    print(
        f"Mask: "
        f"{mask_path.name if mask_path else 'NOT FOUND'}"
    )


    # ========================================================
    # 17. HANDLE MISSING FILES
    # ========================================================

    if ct_path is None or mask_path is None:

        feature_results.append(
            {
                "Patient_ID": patient_id,
                "CT_Found": ct_found,
                "Mask_Found": mask_found,
                "Shape_Match": np.nan,
                "Affine_Match": np.nan,
                "Hemorrhage_Voxels": np.nan,
                "Hemorrhage_Volume_mL": np.nan,
                "Hemorrhage_Slices": np.nan,
                "Mean_Hemorrhage_HU": np.nan,
                "Median_Hemorrhage_HU": np.nan,
                "Std_Hemorrhage_HU": np.nan,
                "Min_Hemorrhage_HU": np.nan,
                "Max_Hemorrhage_HU": np.nan
            }
        )

        print()

        continue


    # ========================================================
    # 18. LOAD CT AND MASK
    # ========================================================

    ct_image = nib.load(
        str(ct_path)
    )

    mask_image = nib.load(
        str(mask_path)
    )


    ct_data = ct_image.get_fdata()

    mask_data = mask_image.get_fdata()


    # ========================================================
    # 19. CHECK SHAPE
    # ========================================================

    shape_match = (
        ct_data.shape
        ==
        mask_data.shape
    )


    # ========================================================
    # 20. CHECK AFFINE
    # ========================================================

    affine_match = np.allclose(
        ct_image.affine,
        mask_image.affine,
        atol=1e-5
    )


    print(
        f"Shape match: "
        f"{shape_match}"
    )

    print(
        f"Affine match: "
        f"{affine_match}"
    )


    # ========================================================
    # 21. STOP IF SHAPES DO NOT MATCH
    # ========================================================

    if not shape_match:

        feature_results.append(
            {
                "Patient_ID": patient_id,
                "CT_Found": 1,
                "Mask_Found": 1,
                "Shape_Match": 0,
                "Affine_Match": int(
                    affine_match
                ),
                "Hemorrhage_Voxels": np.nan,
                "Hemorrhage_Volume_mL": np.nan,
                "Hemorrhage_Slices": np.nan,
                "Mean_Hemorrhage_HU": np.nan,
                "Median_Hemorrhage_HU": np.nan,
                "Std_Hemorrhage_HU": np.nan,
                "Min_Hemorrhage_HU": np.nan,
                "Max_Hemorrhage_HU": np.nan
            }
        )

        print()

        continue


    # ========================================================
    # 22. BINARY HEMORRHAGE MASK
    # ========================================================

    binary_mask = (
        mask_data > 0
    )


    # ========================================================
    # 23. HEMORRHAGE VOXEL COUNT
    # ========================================================

    hemorrhage_voxels = int(
        np.sum(binary_mask)
    )


    # ========================================================
    # 24. VOXEL SIZE
    # ========================================================

    voxel_size = (
        ct_image.header.get_zooms()[:3]
    )


    voxel_volume_mm3 = (
        float(voxel_size[0])
        *
        float(voxel_size[1])
        *
        float(voxel_size[2])
    )


    # ========================================================
    # 25. HEMORRHAGE VOLUME
    # ========================================================

    hemorrhage_volume_mm3 = (
        hemorrhage_voxels
        *
        voxel_volume_mm3
    )


    hemorrhage_volume_ml = (
        hemorrhage_volume_mm3
        /
        1000.0
    )


    # ========================================================
    # 26. HEMORRHAGE-POSITIVE SLICES
    # ========================================================

    voxels_per_slice = np.sum(
        binary_mask,
        axis=(0, 1)
    )


    hemorrhage_slices = int(
        np.sum(
            voxels_per_slice > 0
        )
    )


    # ========================================================
    # 27. HU VALUES INSIDE HEMORRHAGE
    # ========================================================

    if hemorrhage_voxels > 0:

        hemorrhage_hu = ct_data[
            binary_mask
        ]


        hemorrhage_hu = hemorrhage_hu[
            np.isfinite(
                hemorrhage_hu
            )
        ]


        mean_hu = float(
            np.mean(
                hemorrhage_hu
            )
        )


        median_hu = float(
            np.median(
                hemorrhage_hu
            )
        )


        std_hu = float(
            np.std(
                hemorrhage_hu
            )
        )


        min_hu = float(
            np.min(
                hemorrhage_hu
            )
        )


        max_hu = float(
            np.max(
                hemorrhage_hu
            )
        )


    else:

        mean_hu = np.nan

        median_hu = np.nan

        std_hu = np.nan

        min_hu = np.nan

        max_hu = np.nan


    # ========================================================
    # 28. PRINT RESULTS
    # ========================================================

    print(
        f"Hemorrhage voxels: "
        f"{hemorrhage_voxels}"
    )

    print(
        f"Hemorrhage volume: "
        f"{hemorrhage_volume_ml:.2f} mL"
    )

    print(
        f"Hemorrhage-positive slices: "
        f"{hemorrhage_slices}"
    )


    if hemorrhage_voxels > 0:

        print(
            f"Mean hemorrhage HU: "
            f"{mean_hu:.2f}"
        )

        print(
            f"Median hemorrhage HU: "
            f"{median_hu:.2f}"
        )

        print(
            f"HU standard deviation: "
            f"{std_hu:.2f}"
        )


    else:

        print(
            "No hemorrhage detected in mask."
        )


    print()


    # ========================================================
    # 29. SAVE PATIENT FEATURES
    # ========================================================

    feature_results.append(
        {
            "Patient_ID": patient_id,

            "CT_Found": 1,

            "Mask_Found": 1,

            "Shape_Match": int(
                shape_match
            ),

            "Affine_Match": int(
                affine_match
            ),

            "Hemorrhage_Voxels":
                hemorrhage_voxels,

            "Hemorrhage_Volume_mL":
                hemorrhage_volume_ml,

            "Hemorrhage_Slices":
                hemorrhage_slices,

            "Mean_Hemorrhage_HU":
                mean_hu,

            "Median_Hemorrhage_HU":
                median_hu,

            "Std_Hemorrhage_HU":
                std_hu,

            "Min_Hemorrhage_HU":
                min_hu,

            "Max_Hemorrhage_HU":
                max_hu
        }
    )


# ============================================================
# 30. CREATE FEATURE DATAFRAME
# ============================================================

features = pd.DataFrame(
    feature_results
)


# ============================================================
# 31. MERGE PATIENT INFORMATION AND FEATURES
# ============================================================

final_table = patients.merge(
    features,
    on="Patient_ID",
    how="left"
)


# ============================================================
# 32. SAVE FINAL TABLE
# ============================================================

final_table.to_csv(
    output_file,
    index=False
)


# ============================================================
# 33. FINAL SUMMARY
# ============================================================

print()
print("========================================")
print("FINAL SUMMARY")
print("========================================")


print(
    f"Patients processed: "
    f"{len(final_table)}"
)


print(
    f"Male patients: "
    f"{(final_table['Gender'] == 'Male').sum()}"
)


print(
    f"Female patients: "
    f"{(final_table['Gender'] == 'Female').sum()}"
)


print(
    f"ICH-positive patients: "
    f"{final_table['ICH'].sum()}"
)


print(
    f"ICH-negative patients: "
    f"{(final_table['ICH'] == 0).sum()}"
)


positive_masks = int(
    (
        final_table[
            "Hemorrhage_Voxels"
        ] > 0
    ).sum()
)


zero_masks = int(
    (
        final_table[
            "Hemorrhage_Voxels"
        ] == 0
    ).sum()
)


print(
    f"Patients with non-zero mask: "
    f"{positive_masks}"
)


print(
    f"Patients with zero mask: "
    f"{zero_masks}"
)


print()


# ============================================================
# 34. AGE QUALITY CONTROL
# ============================================================

print("========================================")
print("AGE QC")
print("========================================")


age_review = final_table[
    final_table["Age_QC"] != "OK"
]


if len(age_review) > 0:

    print(
        age_review[
            [
                "Patient_ID",
                "Age_years",
                "Gender",
                "Age_QC"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "No unusual age values detected."
    )


# ============================================================
# 35. FEATURE SUMMARY
# ============================================================

print()
print("========================================")
print("FEATURE SUMMARY")
print("========================================")


print(
    final_table[
        [
            "Hemorrhage_Volume_mL",
            "Hemorrhage_Slices",
            "Mean_Hemorrhage_HU",
            "Median_Hemorrhage_HU"
        ]
    ].describe()
)


# ============================================================
# 36. OUTPUT LOCATION
# ============================================================

print()
print("========================================")
print("FILE SAVED")
print("========================================")

print(output_file)