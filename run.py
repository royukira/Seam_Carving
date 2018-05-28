import Seam_Carving

def image_resize_without_mask(filename_input, filename_output, new_height, new_width):
    print("--> Mode: No mask, Resize")
    obj = Seam_Carving.seamCarving(filename_input, new_height, new_width)
    isdone = obj.retargeting()
    if isdone:
        obj.save_img(filename_output)
    else:
        pass


def image_resize_with_mask(filename_input, filename_output, new_height, new_width, filename_mask):
    obj = Seam_Carving.seamCarving(filename_input, new_height, new_width, protect_mask=filename_mask, blur=True)
    isdone = obj.retargeting()
    if isdone:
        obj.save_img(filename_output)
    else:
        pass
    obj.save_img(filename_output)


def object_removal(filename_input, filename_output, omask, pmask=None):
    obj = Seam_Carving.seamCarving(filename_input, 0, 0, obj_mask=omask, protect_mask=pmask, blur=True)
    obj.remove_object()
    obj.save_img(filename_output)



if __name__ == '__main__':
    """
    Put image in in/images folder and protect or object mask in in/masks folder
    Ouput image will be saved to out/images folder with filename_output
    """

    new_width = 938
    new_height = 894

    input_image = '/Users/roy/Documents/GitHub/VisionExperience/Seam_Carving/image/test11.jpg'
    #input_obj_mask = '/Users/roy/Documents/GitHub/VisionExperience/Seam_Carving/image/test9_omask.png'
    #input_protect_mask = '/Users/roy/Documents/GitHub/VisionExperience/Seam_Carving/image/test6_pmask.png'
    output_image = '/Users/roy/Documents/GitHub/VisionExperience/Seam_Carving/image/resize_test11_face.jpg'

    image_resize_without_mask(input_image, output_image, new_height, new_width)
    #image_resize_with_mask(input_image, output_image, new_height, new_width, input_protect_mask)
    #object_removal(input_image, output_image, omask=input_obj_mask, pmask=None)