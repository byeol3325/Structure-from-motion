이 코드들은 SFM(Structure from motion) with no bundle (bundle 관련 코드들은 주석 처리) 코드 입니다.

CMakeFiles.zip을 해당 directory에 푼 다음, 연속된 여러 이미지들(2 ~ @)을 testdata라는 파일에 넣고 코드가 있는 파일과 같은 directory에 넣으면 코드를 돌릴 수 있습니다.

최종 결과는 .ply 파일로 meshlab으로 열면 3d point들을 볼 수 있습니다.

초록색 point들은 camera view point들이고 빨간색 point는 world camera view point입니다.

*** 정량적으로 3d point들을 구하는 방법이므로 camera intrinsic matrix 이 필요합니다.
*** camera intrinsic matrix는 sfm.cpp 파일 가장 위에 입력하면 됩니다.







These codes are SFM(Structure from motion) with no bundle (bundle related codes are commented out).

After unpacking CMakeFiles.zip into the corresponding directory, you can run the code by putting several consecutive images (2 ~ @) in a file called testdata and putting them in the same directory as the file containing the code.

The final result is a .ply file that you can open with meshlab to see the 3d points.

The green points are the camera view points and the red point are the world camera view point.

*** It is a method of quantitatively obtaining 3d points, so camera intrinsic metric is required.
*** You must enter the camera intrinsic matrix at the top of the sfm.cpp file.



input data : several consecutive images (with camera intrinsic matrix)
output data : object 3d points, camera view points
![outout_example](https://user-images.githubusercontent.com/34564290/107140501-ba1ce000-6965-11eb-8bbd-0ca33ec72701.JPG)
