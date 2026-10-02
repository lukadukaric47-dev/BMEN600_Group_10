# BMEN600_Group_10

# BMEN 600 Project
## Team Name
Group 10

## Team Members Present
Luka Dukaric, Javeria Rafique, Dominic Daigle, Mohammadreza Roohitargh

## Candidate Project 1
### Biomedical Problem
Intracranial hemorrhage (ICH) has been implicated in brain midline shift. Early, accurate identification of CT scan structural changes could, therefore, benefit patient outcomes and assessment of the severity of the emergency. As a result, analyzing the relationship between CT image features and the midline shift and whether biological sex differences influences these imaging features would be an important way to contribute to improving clinical care decision-making.

### Possible Research Question
Do CT scan intensity features (i.e. Mean Hounsfield Unit (HU), Asymmetry Index) differ between patients with and without midline shift? How do sex differences contribute to this difference?

### Dataset
The dataset we've selected is: Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation (v1.3.1), published by Murtadha Hssayeni on PhysioNet (2020). https://physionet.org/content/ct-ich/1.3.1/ 

### Biggest Uncertainty
Our dataset of choice is restricted access, however getting availability is fairly simple (make an account and sign the data use agreement). The small sample size (82 total, 36 hemmorhage-positive and 46 hemmorhage-negative) could limit statistical power.

## Candidate Project 2
### Biomedical Problem
Aging is associated with changes in autonomic cardiovascular regulation, which can affect heart-rate variability (HRV). Understanding how HRV changes across the lifespan may help characterize normal cardiovascular aging and provide a baseline for identifying abnormal autonomic function.
### Possible Research Question
How do heart-rate variability features change with age in healthy adults?
### Dataset
https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/
### Biggest Uncertainty
Our biggest uncertainty is determining which HRV features are most appropriate to analyze and whether the ECG recordings require substantial preprocessing before reliable HRV measurements can be extracted.

## Project Decision
We have decided to move forward with Candidate Project 1.
Because: Our team is more familiar with the process of CT, and feel as though this project is more feasible in the time we have during this class. 

## Section 6 – Teamwork and Project Plan
Provide a short plan or timeline of the major milestones from the midterm report to project completion and identify the team member(s) responsible for each.
Literature Review: Everyone will contribute to some extent so that we are all aware of the background and motivation behind this project and can all speak to feasibility of this study. As Dominic has worked with CT scans, he will take the lead on the introduction and background, finding the relevant literature to support our research question. The rest of the team will help identify gaps in the literature to strengthen the motivation section.
Machine Learning (ML): For this project, Mohammadreza will be the team lead for the ML component, ensuring that we are able to distinguish the midline versus non-midline presenting data. As well, he will extract important features (i.e. Mean HU, asymmetry index) for our later sex difference analysis. Other team members will focus on checking and debugging the code, ensuring that the ML workflow is progressing. 
Statistical Analysis: Luka is proficient in MATLAB and R, making him the natural team lead for the statistical analysis portion of the project. He will run ANOVA tests to compare the midline vs. non-midline groups and compare sex differences. The rest of the team will check over the statistic outputs and compare with previous literature to ensure the validity of the findings.
Paper Writing: Dominic will polish up the introduction, motivation, background, and research question sections. Javeria will ensure that the dataset and responsible use section is completed, along with the teamwork and project plan. Luka and Mohammadreza will work in tandem to outline the proposed methods and evaluation and preliminary results.

