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
