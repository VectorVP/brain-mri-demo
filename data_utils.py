 import os
 import nilearn

 from nilearn import plotting


 def visualize_data(data_dir, nii_file_path):
     img = nilearn.image.load_img(os.path.join(data_dir, nii_file_path))
     img_array = nilearn.image.get_data(img)
     file_name = nii_file_path[:-4]
     plotting.plot_anat(img, output_file=f'{file_name}.png')
     print(f'Shape of {nii_file_path}: ', img_array.shape)
