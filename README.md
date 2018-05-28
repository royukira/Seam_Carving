# Seam_Carving
## What is seam carving
* Seam Carving is a `content-aware image` resizing algorithm. 
* The algorithm is for finding an optimal strategy to preserve energy of the image that would be to remove the pixels with `lowest energy` in ascending order (i.e. keep pixels with high energy value). In doing this, we can protect the important content (high energy region) of the image.

## How to find the lowest energy seam?
1. Calculate energy value of every (residual) pixels in the image. 
  * Function: calc_energy()

2. Calculate the cumulative minimum energy by traversing the image from the second row to the last row (Note: enlarge is inverse)
  * Function: calc_CME_forward() `// for removal`; calc_CME_backward() `// for enlarge`

3. The minimum value of the last row in M will indicate the end of the minimal connected vertical seam. Hence, in the final step we backtrack from this minimum entry on M to find the path of the optimal seam 
  * Function: find_seam()

4. Repeat the above steps until resizing is finished (that means we have to loop the steps delta_row or/and delta_col times)

## The main application of Seam Carving in this project:
* Content-aware retargeting // retargeting()
* Remove the marked object //  remove_object()

### Retargeting
1. The scaling up and scaling down functions will be realized in this function
2. In this function, carving all vertical seams firstly and then carving all horizontal seams
`Note: Do not use transport map T !!!`

* First, check if the delta_row = self.output_height - self.input_height is larger than 0
  * yes, scaling up on the vertical direction
  * no, check if the delta_row is smaller than 0
    * yes, scaling down on the vertical direction
    * no, i.e. delta_row = 0; there is no any change on the vertical direction

* Second, check if the delta_col = self.output_width - self.input_width is larger than 0
  * yes, scaling up on the horizontal direction
  * no, check if the delta_col is smaller than 0
    * yes, scaling down on the horizontal direction
    * no, i.e. delta_col = 0; there is no any change on the horizontal direction
    
`NOTE:`
* The function processes the vertical direction firstly, then processes the horizontal direction
* For convenience, the image will be rotated 90 degree before the process of operation on the horizontal direction
* However, because of the limitation of the enlarge step (enlarge an image by k, we find the first k seams for removal, and duplicate them), we can not enlarge the image to the double time

Display:
* Original pic

![](https://github.com/royukira/Seam_Carving/blob/master/photo/normal_vs_seam/test8.jpg)

* Seam carving

![](https://github.com/royukira/Seam_Carving/blob/master/photo/normal_vs_seam/output_test8_withoutmask_400_183.jpg)

* Normal retargeting

![](https://github.com/royukira/Seam_Carving/blob/master/photo/normal_vs_seam/ns_test8_400_183.jpg)

* Energy Map

![](https://github.com/royukira/Seam_Carving/blob/master/photo/normal_vs_seam/up_energy_map_test8.jpg)

### Remove the marked object
Using the principle of the algorithm. The marked region will be assign very low energy values so that can remove the marked object (marked by black)

`Display:`
* Original pic

![](https://github.com/royukira/Seam_Carving/blob/master/photo/object_removal/test9.jpg)

* Without protected mask
  * result
 
 ![](https://github.com/royukira/Seam_Carving/blob/master/photo/object_removal/objRemove_test9_test_bad.jpg)

  * Energy map
 
 ![](https://github.com/royukira/Seam_Carving/blob/master/photo/object_removal/up_energy_map0.jpg)
 
* With protected mask
  * result
 
 ![](https://github.com/royukira/Seam_Carving/blob/master/photo/object_removal/objRemove_test9_mask.jpg)
 
  * Energy map
 
 ![](https://github.com/royukira/Seam_Carving/blob/master/photo/object_removal/up_energy_map_mask0.jpg)
 
### Mask

* Protected mask
 * For protecting some important content, sometimes we need to mark them manually for the better resizing effect.
 * The marked image is called protect-mask
 * The marked region will be assign very high energy values so that can prevent the content in marked region from mistaken removal
 
 * Protected marking
 
 ![](https://github.com/royukira/Seam_Carving/blob/master/photo/mask_vs_without_mask/test6_pmask.png)
 
 * Without protected mask
 
 ![](https://github.com/royukira/Seam_Carving/blob/master/photo/mask_vs_without_mask/output_test6_test_without_mask.jpg)
 
 * With protected mask
 
 ![](https://github.com/royukira/Seam_Carving/blob/master/photo/mask_vs_without_mask/output_test6_mask_1.jpg)


* Objected mask
 * Mark the removal objective
 * The marked region will be assign very low energy values (negtive value)
 
 * Removal objective
 
 ![](https://github.com/royukira/Seam_Carving/blob/master/photo/object_removal/test9_omask.png)
 
### Face detection
In this implementation, I use `Haar Cascades classifier` based on the `Viola–Jones` algorithm. Before processing the image, the face detection can recognize human faces and automatically generate protected masks for those faces.

`Display:`

* Face detection 

![](https://github.com/royukira/Seam_Carving/blob/master/photo/face_detection/face_detect.png)
![](https://github.com/royukira/Seam_Carving/blob/master/photo/face_detection/test12.png)

* Protected mask

![](https://github.com/royukira/Seam_Carving/blob/master/photo/face_detection/Figure_1.png)
![](https://github.com/royukira/Seam_Carving/blob/master/photo/face_detection/auto_protect_mask.png)

* Result

 * `BAD`
 ![](https://github.com/royukira/Seam_Carving/blob/master/photo/face_detection/resize_test11_nonFace.jpg)
 ![](https://github.com/royukira/Seam_Carving/blob/master/photo/face_detection/resize_test12_bad.jpg)

 * `GOOD`
 ![](https://github.com/royukira/Seam_Carving/blob/master/photo/face_detection/resize_test11_face.jpg)
 ![](https://github.com/royukira/Seam_Carving/blob/master/photo/face_detection/resize_test12_face_detect.jpg)

* Energy map

 * `Without face detection`
 ![](https://github.com/royukira/Seam_Carving/blob/master/photo/face_detection/down_energy_map1.jpg)
 
 * `With face detection`
 ![](https://github.com/royukira/Seam_Carving/blob/master/photo/face_detection/down_energy_map_mask1.jpg)
