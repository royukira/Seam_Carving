
import numpy as np
import cv2
import ShowProcess
from matplotlib import pyplot as plt


class seamCarving:

    def __init__(self, src, output_height, output_width, protect_mask=None, obj_mask=None, blur=False, face_detect=False):
        # read the input image
        self.input_image = cv2.imread(src).astype(np.float64)
        self.gray_input = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGRA2GRAY)
        if blur:
            self.input_image = self.blur_filter(self.input_image)  # Blur the image
        self.input_height, self.input_width = self.input_image.shape[: 2]

        # the size of output image
        self.output_width = output_width
        self.output_height = output_height

        # Check whether there is a protect mask
        # The protect mask is used for protecting object from distortion or destruction caused by re-targeting
        if protect_mask is not None:
            self.pmask = cv2.imread(protect_mask, 0).astype(np.float64)
            self.pmask = self.make_mask(self.pmask)
            #plt.imshow(self.pmask)  # for test
            #plt.show()
        else:
            if face_detect:
                # The face detection will create a protect region for the face
                self.face_detection(self.gray_input)
            else:
                self.pmask = None
        # Check whether there is a object mask
        # The object mask is used for object removal
        if obj_mask is not None:
            self.omask = cv2.imread(obj_mask, 0).astype(np.float64)
            self.omask = self.make_mask(self.omask)
            #plt.imshow(self.omask)  # for test
            #plt.show()
        else:
            self.omask = None

        # kernel for calculating the energy map
        self.kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        self.kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
        self.kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)

        # Convert input image to gray scale
        #self.gray_input = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)

        # output image
        self.output_image = np.copy(self.input_image)

        # Constant
        # the region of protection will be protected by a big positive value which make the energy value high
        # on the other hand, the region of object that need to be removed will be mark with a very big negative value
        # Thus, protection -- the pixel in protection region * self.C
        #       remove object -- the pixel of object * (-self.C)
        self.C = 10000

        # sign of rotation
        self.rotated = False  # if True, the image is rotated

# ======================== Pre-process =========================

    def make_mask(self, mask):
        output_mask = np.zeros(mask.shape[:2], np.uint8)
        output_mask[mask != 0] = 0
        output_mask[mask == 0] = 1
        return output_mask

    def blur_filter(self,img):
        output = cv2.blur(img, (3,3))
        return output

# ======================== The main function =========================
# ======================== 1.Re-targeting     ========================
# ========================   1.1 Scaling up   ========================
# ========================   1.2 Scaling down ========================
# ======================== 2.Remove the marked object ================

    def retargeting(self):
        """
        The scaling up and scaling down functions will be realized in this function

        In this function, carving all vertical seams firstly and then carving all horizontal seams

        Note: Do not use transport map T !!!

        - First, check if the delta_row = self.output_height - self.input_height is larger than 0
            -- yes, scaling up on the vertical direction
            -- no, check if the delta_row is smaller than 0
                -- yes, scaling down on the vertical direction
                -- no, i.e. delta_row = 0; there is no any change on the vertical direction

        - Second, check if the delta_col = self.output_width - self.input_width is larger than 0
            -- yes, scaling up on the horizontal direction
            -- no, check if the delta_col is smaller than 0
                -- yes, scaling down on the horizontal direction
                -- no, i.e. delta_col = 0; there is no any change on the horizontal direction

        Overall, this function can satisfy resizing whatever the size of the new image is. The function processes
        the vertical direction firstly, then processes the horizontal direction

        For convenience, the image will be rotated 90 degree before the process of operation on the horizontal direction

        :return:
        """
        delta_row = int(self.output_height - self.input_height)
        delta_col = int(self.output_width - self.input_width)

        # No change
        if delta_row == 0 and delta_col == 0:
            print("--> The size of output image is same as the input image... ")
            return False

        # vertical first
        if delta_col > 0:
            print("--> Start to scaling up on the vertical direction")
            self.scaling_up(delta_col)
        elif delta_col < 0:
            print("--> Start to scaling down on the vertical direction")
            self.scaling_down(delta_col * -1)

        # the next: horizontal
        if delta_row > 0:
            self.output_image = self.rotate_img(self.output_image)  # rotate the img first
            if self.pmask is not None:
                self.pmask = self.rotate_mask(self.pmask)
            print("--> Start to scaling up on the horizontal direction")
            self.scaling_up(delta_row)
        elif delta_row < 0:
            self.output_image = self.rotate_img(self.output_image)  # rotate the img first
            if self.pmask is not None:
                self.pmask = self.rotate_mask(self.pmask)
            print("--> Start to scaling down on the horizontal direction")
            self.scaling_down(delta_row * -1)

        # rotate back
        if self.rotated:
            self.output_image = self.re_rotate_img(self.output_image)
            if self.pmask is not None:
                self.pmask = self.re_rotate_mask(self.pmask)

        print("--> Resizing is DONE!")
        return True

    def remove_object(self):
        """
        Remove the marked object
        :return:
        """
        if self.omask is None:
            print("--> Please mark objects using mask")

        obj_height, obj_width = self.get_obj_size()

        if obj_height < obj_width:
            self.output_image = self.rotate_img(self.input_image)
            self.omask = self.rotate_mask(self.omask)

            if self.pmask is not None:
                self.pmask = self.rotate_mask(self.pmask)

        count = 0

        print("--> Start to remove the object...")
        while len(np.where(self.omask[:, :] > 0)[0]) > 0:
            # Step 1: calculate the energy value of a pixel
            energy_map = self.calc_energy()
            energy_map[np.where(self.omask[:, :] > 0)] *= -self.C
            if self.pmask is not None:
                energy_map[np.where(self.pmask[:, :] > 0)] *= self.C
            # Step 2: calculate cumulative minimum energy
            cme_map = self.calc_CME_forward(energy_map)

            # Step 3: find the seams
            seam_idx = self.find_seam(cme_map)

            # Step 4: remove the seams
            if count % 20 == 0:  # save the seam plot every 20 steps
                self.seam_removal(seam_idx, name='remove_obj_seam{0}'.format(count))
            else:
                self.seam_removal(seam_idx)
            self.omask = self.seam_removal_mask(seam_idx, self.omask)
            if self.pmask is not None:
                self.pmask = self.seam_removal_mask(seam_idx, self.pmask)
            count += 1

        if self.rotated:
            delta_pix = self.input_height - self.output_image.shape[1]
        else:
            delta_pix = self.input_width - self.output_image.shape[1]

        # Compensate
        self.scaling_up(delta_pix)  # scaling up to the original size

        if self.rotated:
            self.output_image = self.re_rotate_img(self.output_image)


# ======================== Process ===============================
# ======================== 1.Scaling up ==========================
# ======================== 2.Scaling down ========================

    def scaling_up(self, delta_pixel):
        """
        Insert some seams with low energy

        To achieve effective enlarging, it is important to balance between the original image content and the
        artificially inserted parts. Therefore, to enlarge an image by k, we find the first k seams for removal,
        and duplicate them in order

        1. Delete k seams
        2. Duplicate them in order

        :param delta_pixel:
        :return:
        """
        processBar = ShowProcess.ShowProcess(delta_pixel - 1)

        if self.pmask is not None:
            temp_image = np.copy(self.output_image)
            temp_mask = np.copy(self.pmask)
            seams_record = []

            print("--> Scaling up with mask * Step 1: Remove {0} seams".format(delta_pixel))
            for dummy in range(delta_pixel):
                energy_map = self.calc_energy()
                energy_map[np.where(self.pmask[:, :] > 0)] *= self.C
                cumulative_map = self.calc_CME_backward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                seams_record.append(seam_idx)
                if dummy % 20 == 0 or dummy == delta_pixel - 1:
                    self.seam_removal(seam_idx, name='seam_mask{0}'.format(dummy))
                else:
                    self.seam_removal(seam_idx)
                self.pmask = self.seam_removal_mask(seam_idx, self.pmask)
                processBar.show_process()

                if dummy % 20 == 0:
                    self.plot_energy_map(energy_map, name='up_energy_map_mask{0}'.format(dummy))

            processBar.close(words='--> Scaling up with mask * Step 1 is DONE!')

            self.output_image = np.copy(temp_image)
            self.pmask = np.copy(temp_mask)

            n = len(seams_record)
            for dummy in range(n):
                seam = seams_record.pop(0)
                self.seam_insertion(seam)
                self.pmask = self.seam_insertion_mask(seam, self.pmask)
                seams_record = self.seam_update(seams_record, seam)
                processBar.show_process(dummy)
            processBar.close(words='--> Scaling up with mask * Step 2 is DONE!')

        else:
            """ Without mask """
            temp_image = np.copy(self.output_image)
            seams_record = []
            print("--> Scaling up * Step 1: Remove {0} seams".format(delta_pixel))
            for dummy in range(delta_pixel):
                energy_map = self.calc_energy()
                cumulative_map = self.calc_CME_backward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                seams_record.append(seam_idx)
                if dummy % 20 == 0 or dummy == delta_pixel - 1:
                    self.seam_removal(seam_idx, name='seam{0}'.format(dummy))
                else:
                    self.seam_removal(seam_idx)
                processBar.show_process()  # only display on terminal

                if dummy % 20 == 0 or dummy == delta_pixel-1:
                    self.plot_energy_map(energy_map,name='up_energy_map{0}'.format(dummy))

            processBar.close(words='--> Scaling up * Step 1 is DONE!')

            self.output_image = np.copy(temp_image)

            n = len(seams_record)

            print("--> Scaling up * Step 2: Duplicate {0} seams".format(delta_pixel))
            for dummy in range(n):
                seam = seams_record.pop(0)
                self.seam_insertion(seam)
                seams_record = self.seam_update(seams_record, seam)
                processBar.show_process(dummy)
            processBar.close(words='--> Scaling up * Step 2 is DONE!')

    def scaling_down(self, delta_pixel):
        processBar = ShowProcess.ShowProcess(delta_pixel - 1)

        if self.pmask is not None:
            print("--> Scaling down with protect mask: Remove {0} seams".format(delta_pixel))
            for dummy in range(delta_pixel):
                energy_map = self.calc_energy()
                energy_map[np.where(self.pmask > 0)] *= self.C  # 只增大白色的区域
                cumulative_map = self.calc_CME_forward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                if dummy % 20 == 0 or dummy == delta_pixel - 1:
                    self.seam_removal(seam_idx, name='seam{0}'.format(dummy))
                else:
                    self.seam_removal(seam_idx)
                self.pmask = self.seam_removal_mask(seam_idx,self.pmask)
                processBar.show_process(dummy)

                if dummy % 20 == 0 or dummy == delta_pixel-1:
                    self.plot_energy_map(energy_map, name='down_energy_map_mask{0}'.format(dummy))

            processBar.close(words='--> Scaling down with protect mask is DONE!')

        else:
            # No protect mask
            print("--> Scaling down without protect mask: Remove {0} seams".format(delta_pixel))
            for dummy in range(delta_pixel):
                energy_map = self.calc_energy()
                cumulative_map = self.calc_CME_forward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                if dummy % 20 == 0 or dummy == delta_pixel - 1:
                    self.seam_removal(seam_idx, name='seam{0}'.format(dummy))
                else:
                    self.seam_removal(seam_idx)
                processBar.show_process(dummy)

                if dummy % 20 == 0 or dummy == delta_pixel-1:
                    self.plot_energy_map(energy_map,name='down_energy_map{0}'.format(dummy))

            processBar.close(words='--> Scaling down without protect mask is DONE!')


# ======================== Steps =====================================
# =============== 1. calculate the energy of a pixel =================
# =============== 2. calculate cumulative minimum energy =============
# =============== 3. backtrack the seam path =========================
# =============== 4. remove the seam =================================
# =============== 5. (for enlarge) seam insertion ====================

    def calc_energy(self):
        """
        Calculate energy of every pixel (R,G,B)

        Use the OpenCV function Scharr to calculate a more accurate derivative for a kernel of size 3-by-3

        gradient = eb + eg + er

        :return:
        """
        b, g, r = cv2.split(self.output_image)
        eb = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        eg = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        er = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))

        g = eb + eg + er

        return g

    def calc_CME_forward(self, energy_map):
        """
        For seam removal
        :return:
        """
        matrix_x = self.neighbor_filter(self.kernel_x)
        matrix_y_left = self.neighbor_filter(self.kernel_y_left)
        matrix_y_right = self.neighbor_filter(self.kernel_y_right)

        m, n = energy_map.shape
        cme = np.copy(energy_map)
        for row in range(1, m):
            for col in range(n):
                # Special cases: pixels are far left or far right
                if col == 0:
                    top_right = cme[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    up = cme[row - 1, col] + matrix_x[row - 1, col]
                    # CME
                    cme[row, col] = energy_map[row, col] + min(top_right, up)
                elif col == n - 1:
                    top_left = cme[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    up = cme[row - 1, col] + matrix_x[row - 1, col]
                    # CME
                    cme[row, col] = energy_map[row, col] + min(top_left, up)
                # Normal cases
                else:
                    top_left = cme[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    top_right = cme[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    up = cme[row - 1, col] + matrix_x[row - 1, col]
                    # CME
                    cme[row, col] = energy_map[row, col] + min(top_left, top_right, up)
        return cme

    def calc_CME_backward(self, energy_map):
        """
        For enlarge, the step is inverse version of the scaling down
        :return:
        """
        m, n = energy_map.shape
        cme = np.copy(energy_map)
        for row in range(1, m):  # start from the second row
            for col in range(n):
                cme[row, col] = energy_map[row, col] + np.amin(cme[row - 1, max(col - 1, 0): min(col + 2, n - 1)])
        return cme

    def neighbor_filter(self, kernel):
        b, g, r = cv2.split(self.output_image)
        output = np.absolute(cv2.filter2D(b, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(g, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(r, -1, kernel=kernel))
        return output

    def find_seam(self, cme):
        """
        Backtrack and find the seam
        :return:
        """
        m, n = cme.shape
        seams = np.zeros((m,), dtype=np.uint32)
        seams[-1] = np.argmin(cme[-1])
        for row in range(m - 2, -1, -1):
            prv_x = seams[row + 1]
            if prv_x == 0:
                seams[row] = np.argmin(cme[row, : 2])
            else:
                seams[row] = np.argmin(cme[row, prv_x - 1: min(prv_x + 2, n - 1)]) + prv_x - 1
        return seams

    def seam_removal(self, remove_pix, name=None):
        """
        remove the seams
        :return:
        """
        if name is not None:
            if self.rotated:
                tmp = self.re_rotate_img(self.output_image)
                self.rotated = True
                self.plot_seam(remove_pix, tmp, name=name)
            else:
                self.plot_seam(remove_pix, self.output_image, name=name)

        m, n = self.output_image.shape[: 2]
        output = np.zeros((m, n - 1, 3))
        for row in range(m):
            col = remove_pix[row]
            output[row, :, 0] = np.delete(self.output_image[row, :, 0], [col])
            output[row, :, 1] = np.delete(self.output_image[row, :, 1], [col])
            output[row, :, 2] = np.delete(self.output_image[row, :, 2], [col])
        self.output_image = np.copy(output)

    def seam_insertion(self, delta_pix):
        """
        For scaling up, we need to insert some seams with low energy
        :return:
        """
        m, n = self.output_image.shape[: 2]
        output = np.zeros((m, n + 1, 3))
        for row in range(m):
            col = delta_pix[row]
            for ch in range(3):
                if col == 0:
                    p = np.average(self.output_image[row, col: col + 2, ch])
                    output[row, col, ch] = self.output_image[row, col, ch]
                    output[row, col + 1, ch] = p
                    output[row, col + 1:, ch] = self.output_image[row, col:, ch]
                else:
                    p = np.average(self.output_image[row, col - 1: col + 1, ch])
                    output[row, : col, ch] = self.output_image[row, : col, ch]
                    output[row, col, ch] = p
                    output[row, col + 1:, ch] = self.output_image[row, col:, ch]
        self.output_image = np.copy(output)

    def seam_removal_mask(self, remove_pix, mask):
        """
        remove the seams of mask
        :return:
        """
        m, n = mask.shape
        output = np.zeros((m, n - 1))
        for row in range(m):
            col = remove_pix[row]
            output[row, :] = np.delete(mask[row, :], [col])
        mask = np.copy(output)
        return mask

    def seam_insertion_mask(self, delta_pix, mask):
        """
        Insert some seams with low energy of mask
        :return:
        """
        m, n = mask.shape
        output = np.zeros((m, n + 1))
        for row in range(m):
            col = delta_pix[row]
            if col == 0:
                p = np.average(mask[row, col: col + 2])
                output[row, col] = mask[row, col]
                output[row, col + 1] = p
                output[row, col + 1:] = mask[row, col:]
            else:
                p = np.average(mask[row, col - 1: col + 1])
                output[row, : col] = mask[row, : col]
                output[row, col] = p
                output[row, col + 1:] = mask[row, col:]
        mask = np.copy(output)
        return mask

    def seam_update(self, record, update_seam):
        output = []
        for seam in record:
            seam[np.where(seam >= update_seam)] += 2
            output.append(seam)
        return output


# ======================== Rotate ================================
# ======================== 1.Rotate image ========================
# ======================== 2. Rotate mask ========================

    def rotate_img(self, image):
        m, n, ch = image.shape
        rotate_img = np.zeros((n, m, ch))
        image_flip = np.fliplr(image)
        for c in range(ch):
            for row in range(m):
                rotate_img[:, row, c] = image_flip[row, :, c]
        self.rotated = True
        return rotate_img

    def re_rotate_img(self,image):
        m, n, ch = image.shape
        re_rotate_img = np.zeros((n, m, ch))
        for c in range(ch):
            for row in range(m):
                re_rotate_img[:, m - 1 - row, c] = image[row, :, c]
        self.rotated = False
        return re_rotate_img

    def rotate_mask(self, mask):
        m, n = mask.shape
        rotate_mask = np.zeros((n, m))
        image_flip = np.fliplr(mask)
        for row in range(m):
            rotate_mask[:, row] = image_flip[row, :]
        return rotate_mask

    def re_rotate_mask(self, mask):
        m, n = mask.shape
        re_rotate_mask = np.zeros((n, m))
        for row in range(m):
            re_rotate_mask[:, m - 1 - row] = mask[row, :]
        return re_rotate_mask


# ======================== Plot ===================================
# ======================== 1.Plot energy map ======================
# ======================== 2.Plot CME map =========================
# ======================== 3.Plot seam ============================

    def plot_energy_map(self, energy_map, name):
        cv2.imwrite('./Process/{0}.jpg'.format(name), energy_map)
        #print("--> The energy map is saved!")

    def plot_CME_map(self, cme, name):
        cv2.imwrite('./Process/{0}.jpg'.format(name), cme)

    def plot_seam(self, seam_idx, img, name):
        if self.rotated:
            n, m = img.shape[: 2]
        else:
            m, n = img.shape[: 2]
        seam_img = np.copy(img)
        for row in range(m):
            col = seam_idx[row]
            if self.rotated:
                seam_img[col, row, 0] = 0
                seam_img[col, row, 1] = 0
                seam_img[col, row, 2] = 255  # Red
            else:
                seam_img[row, col, 0] = 0
                seam_img[row, col, 1] = 0
                seam_img[row, col, 2] = 255  # Red
        cv2.imwrite('./Process/{0}.jpg'.format(name), seam_img)

# ======================== Others ========================
# ======================== 1.Get object size =============
# ======================== 2.Save output image ===========

    def get_obj_size(self):
        rows, cols = np.where(self.omask > 0)
        height = np.amax(rows) - np.amin(rows) + 1
        width = np.amax(cols) - np.amin(cols) + 1
        return height, width

    def save_img(self, filename):
        cv2.imwrite(filename, self.output_image.astype(np.uint8))
        print("--> The result has been saved!")


# ======================== Assistance ====================
# ======================== 1.Face detection ==============

    def face_detection(self, grayImg):
        face_cascade = cv2.CascadeClassifier(
            './aarcascade_frontalface_alt2.xml')

        faces = face_cascade.detectMultiScale(grayImg, 1.3, 5)

        # the size of the window
        x = faces[0][0]
        y = faces[0][1]
        w = faces[0][2]
        h = faces[0][3]

        self.pmask = np.zeros((self.input_height, self.input_width))

        for xw in range(x, x+w):
            for yh in range(y, y+h):
                self.pmask[yh, xw] = 1

        #plt.imshow(self.pmask)
        #plt.show()







