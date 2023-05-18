"""
This is example of function which loading preprocessed  (cooregistred to MNI 152 template) data and FS masks
"""


def load_data(data_path,subj_id):    
    wm_seg_path =f'{data_path}{subj_id}/result/wmseg.nii.gz'
    aparc_seg_path =f'{data_path}{subj_id}/result/aparcaseg.nii.gz'
    aparc_seg_path2 =f'{data_path}{subj_id}/result/aparc2009aseg.nii.gz'
    brain_path =f'T1_linear_preproc/SRPBS/t1_linear_bfc/sub-{subj_id}/Warped.nii.gz'
    template_path =f'template/Q1-Q6_RelatedParcellation210_AverageT1w_restore.nii.gz'

    brain_file =nib.load(brain_path)
    inv_matrix =inv(brain_file.affine)
    brain_image = np.array(brain_file.dataobj).astype(np.float64)
    
    wm_seg_image, aparcaseg_image, aparcaseg2_image,template_image =None,None,None,None
    if Path(wm_seg_path).exists():
        wm_seg_file =nib.load(wm_seg_path)
        wm_seg_image = np.array(wm_seg_file.dataobj).astype(int)
    if Path(aparc_seg_path).exists(): 
        aparcaseg_file =nib.load(aparc_seg_path)
        aparcaseg_image = np.array(aparcaseg_file.dataobj).astype(int)
    if Path(aparc_seg_path2).exists():    
        aparcaseg_file2 =nib.load(aparc_seg_path2)
        aparcaseg2_image = np.array(aparcaseg_file2.dataobj).astype(int)      
    return brain_image, wm_seg_image, aparcaseg_image, aparcaseg2_image,inv_matrix




class RegionSegmentor():
    
    """
        Proposed Region Segmentor extract intensities from regions specified in csv file.
        
        We apply White Matter, Gray Matter and Subcortical masks received from 
        FreeSurfer segmentation to find intersection of extracted intensity from proposed region and specific mask corresponded to this region.
        
    """
    
    
    def __init__(self, prefix, not_random):
        if not_random:
            self.lipid_regions =pd.read_csv("75_regions_union.csv")
            self.result_subject_data =self.lipid_regions.copy()
        else:
            self.lipid_regions =pd.read_csv("75_random_regions.csv")
            self.result_subject_data =self.lipid_regions.iloc[:, :5].copy()
            
        self.prefix=prefix
        self.stats_path =Path('column_names.txt')
        self.stats_map =flatten_freesurfer_stats(self.stats_path)
        self.stats_map['SegId']=self.stats_map['SegId'].astype(int)
    
    
    def calculate_intensity_intersection_array(self, coords,brain,segmentation):
        brain =brain*segmentation
        intensities =[]
        shifts =[-3,-2,-1,0,1,2,3]
        for i in shifts:
            for j in shifts:
                for k in shifts: 
                     if brain[coords[0]+i,coords[1]+j,coords[2]+k]!=0:
                         intensities.append(brain[coords[0]+i,coords[1]+j,coords[2]+k])
        return np.array(intensities)
    
    def calculate_intensity_array(self, coords, brain):
        intensities =[]
        shifts =[-3,-2,-1,0,1,2,3]
        for i in shifts:
            for j in shifts:
                for k in shifts:  
                     intensities.append(brain[coords[0]+i,coords[1]+j,coords[2]+k])
        return np.array(intensities)
    
    
    def calculate_region(self, coords,tissue_type, brain, segmentations, is_segm_region):
        segment_name,segment_id =None, None
        if segmentations and is_segm_region:
            for segmentation in segmentations:
                if is_segm_region:
                    segment_id =segmentation[coords[0], coords[1], coords[2]]
                    segment_name =self.stats_map.loc[self.stats_map['SegId']==int(segment_id)]['StructName'].values[0]
                    if segment_name =='Unknown':
                        continue
                    else:
                        intensity =self.calculate_intensity_intersection_array(coords, brain, segmentation ==segment_id)
                    return segment_name,segment_id,intensity
        elif segmentations:
            aparcaseg_image, aparcaseg2_image,wm_seg_image = segmentations
            
            if tissue_type == 'GM' and aparcaseg_image is not None:
                intensity =self.calculate_intensity_intersection_array(coords, brain, aparcaseg_image>=1000)
                if len(intensity)>0:
                    segment_name =tissue_type
                    return segment_name,segment_id,intensity 
            elif tissue_type == 'WM' and wm_seg_image is not None:
                intensity =self.calculate_intensity_intersection_array(coords, brain, wm_seg_image>=3000)
                if len(intensity)>0:
                    segment_name =tissue_type
                    return segment_name,segment_id,intensity    
            elif wm_seg_image is not None:
                intensity =self.calculate_intensity_intersection_array(coords, brain, (wm_seg_image<1000) & (wm_seg_image>0) & (wm_seg_image!=2)& (wm_seg_image!=41))
                if len(intensity)>0:
                    segment_name =tissue_type
                    return segment_name,segment_id,intensity
       
        intensity =self.calculate_intensity_array(coords, brain)
        return segment_name,segment_id,intensity
        
    def calculate_intensities(self, brain, segmentations,inv_matrix, subj_id, is_segm_region=False):
        self.result_subject_data['freesurfer_segment_name'] =None
        self.result_subject_data['freesurfer_segment_id'] =None
        self.result_subject_data['intensity'] =None
        for i in range(self.result_subject_data.shape[0]):
            new_coords =nilearn.image.coord_transform(self.result_subject_data.loc[i,'mni_x'],
                                                      self.result_subject_data.loc[i,'mni_y'],
                                                      self.result_subject_data.loc[i,'mni_z'], i
                                                      nv_matrix)
            new_coords =(round(new_coords[0]), round(new_coords[1]), round(new_coords[2]))
            segment_name,segment_id,intensity =self.calculate_region(new_coords,self.result_subject_data.loc[i,'FS'], brain,segmentations,is_segm_region)
            self.result_subject_data.loc[i,'freesurfer_segment_name'] =segment_name
            self.result_subject_data.loc[i,'freesurfer_segment_id'] =segment_id
            self.result_subject_data.loc[i,'intensity'] =str(intensity)
        self.save(subj_id)    
    
    def save(self,subj_id):
        self.result_subject_data.to_csv(Path(self.prefix) / f"{subj_id}.csv")

        
def calculate_intensities(prefix, data_path,not_random=True,is_segm_region=False):
    """
    Function wich extract intensities from refion around particular coordinate and with intersection of  pecific mask corresponded to this region.

    :prefix: output data directory path
    :data_path: input data directory path
    :not_random: choose curent coords or randomly selected from csv file, default =True
    :is_segm_region: is we extract intensity with current region? For example, if we extract coord from amigdala region, we need to intersect it with amigdala mask from fs. Otherwise, we intersect it with WM or GM or SubCort mask.

    """ 
    seg =RegionSegmentor(prefix, not_random)
    for subj_id in os.listdir(data_path):
        try:
            if subj_id not in ['.ipynb_checkpoints/','.ipynb', 'Untitled1.ipynb', 'mni_registration.ipynb']:
                    segmentations =[]
                    brain_image, wm_seg_image, aparcaseg_image, aparcaseg2_image,inv_matrix =load_data(data_path,subj_id)

                    if not is_segm_region:
                        segmentations =[aparcaseg_image, aparcaseg2_image,wm_seg_image]
                    else:
                        for i, elem in enumerate([aparcaseg_image, aparcaseg2_image,wm_seg_image]):
                                if elem is not None:
                                    segmentations.append(elem)
                                else:
                                    ma ={0: "wm_seg_image",
                                         1: "aparcaseg_image",
                                         2: "aparcaseg2_image"}
                                    print("Not found in ", subj_id, ":", ma[i])
                    seg.calculate_intensities(brain_image, segmentations,inv_matrix,subj_id,is_segm_region)
        except Exception as e:
            print(e)
            
if __name__ == "__main__":           
    calculate_intensities(prefix ='intensities/SRPBS_t1_with_segmentation',
                          data_path ='MNI_preproc/',
                          not_random=True,
                          is_segm_region=False)            