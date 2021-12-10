# This script downloads the evaluation set for the 3D Match Geometric Registration 
# benchmark from the official website. 
# Benchmark files and downloaded to the specified directory
#       sh get_3dmatch_eval.sh <dataset_directory>

mkdir 3dmatch_eval_temp
cd 3dmatch_eval_temp

# download
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/7-scenes-redkitchen.zip
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/7-scenes-redkitchen-evaluation.zip
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-home_at-home_at_scan1_2013_jan_1.zip
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-home_at-home_at_scan1_2013_jan_1-evaluation.zip
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-home_md-home_md_scan9_2012_sep_30.zip
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-home_md-home_md_scan9_2012_sep_30-evaluation.zip
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_uc-scan3.zip
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_uc-scan3-evaluation.zip
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_umd-maryland_hotel1.zip
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_umd-maryland_hotel1-evaluation.zip
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_umd-maryland_hotel3.zip
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-hotel_umd-maryland_hotel3-evaluation.zip
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-mit_76_studyroom-76-1studyroom2.zip
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-mit_76_studyroom-76-1studyroom2-evaluation.zip
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika.zip
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/scene-fragments/sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika-evaluation.zip

# unzip and delete zip files
unzip 7-scenes-redkitchen.zip
unzip 7-scenes-redkitchen-evaluation.zip
unzip sun3d-home_at-home_at_scan1_2013_jan_1.zip
unzip sun3d-home_at-home_at_scan1_2013_jan_1-evaluation.zip
unzip sun3d-home_md-home_md_scan9_2012_sep_30.zip
unzip sun3d-home_md-home_md_scan9_2012_sep_30-evaluation.zip
unzip sun3d-hotel_uc-scan3.zip
unzip sun3d-hotel_uc-scan3-evaluation.zip
unzip sun3d-hotel_umd-maryland_hotel1.zip
unzip sun3d-hotel_umd-maryland_hotel1-evaluation.zip
unzip sun3d-hotel_umd-maryland_hotel3.zip
unzip sun3d-hotel_umd-maryland_hotel3-evaluation.zip
unzip sun3d-mit_76_studyroom-76-1studyroom2.zip
unzip sun3d-mit_76_studyroom-76-1studyroom2-evaluation.zip
unzip sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika.zip
unzip sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika-evaluation.zip

rm *.zip

# move file to input location
cd ..
mv 3dmatch_eval_temp $1 
