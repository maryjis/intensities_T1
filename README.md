  The repo helps to extract intensities around particular voxel from T1/T2 MRI data for analysing data (COBRE dataset).
  
  We extract intensities from particular coordinate and +-3 adjacent voxels in each direction ( 7*7*7 voxels cube).
  
  After that we intersect this  voxels cube with masks from White Matter (WM), Gray Matter (GM) and subcortical regions .
 
 
 Proposed Region Segmentor class extract intensities from regions specified in csv file. ( see, 75_region_union.csv). Each coordinate describes in  MNI-152 and Talairach coordinates.
        
 We apply White Matter, Gray Matter and Subcortical masks received from FreeSurfer segmentation.
 
Each T1w/ T2w MRI data was registred to MNI-152 template and bias field corrected.
